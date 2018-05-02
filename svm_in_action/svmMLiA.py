# -*- coding: utf-8 -*-

import random

import numpy as np


def kernelTrans(X, A, kTup):
    """
    kTup - 是一个核函数信息的元组: 第一个参数是用于描述很函数类型的字符串, 其余两个是核函数可能需要的可选参数

    """
    m, n = X.shape
    K = np.mat(np.zeros((m, 1)))
    if kTup[0] == "lin":
        # 线性核函数, 内积计算在"所有数据集"与"数据集的一行"这两个输入之间展开
        K = X * A.T
    elif kTup[0] == "rbf":
        # 径向基核函数, 计算每个元素的高斯函数值, 然后将结果应用到整个向量上.
        for j in range(m):
            deltaRow = X[j, :] - A
            K[j] = deltaRow * deltaRow.T
        K = np.exp(K / (-1 * kTup[1] ** 2))  # np 中, 除法对整个矩阵元素展开计算, 而非 MATLAB 的求矩阵的逆
    else:
        raise NameError("Houston we have a problem -- The Kernel is not recognized")
    return K


def loadDataSet(fileName):
    dataMat = []
    labelMat = []
    with open(fileName) as f:
        for line in f.readlines():
            lineArr = line.strip().split("\t")
            dataMat.append([float(lineArr[0]), float(lineArr[1])])
            labelMat.append(float(lineArr[2]))

    return dataMat, labelMat


def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    elif aj < L:
        aj = L
    return aj


def selectJrand(i, m):
    j = i
    while i == j:
        j = int(random.uniform(0, m))
    return j


class optStruct():
    """封装了一些常用数据"""

    def __init__(self, dataMatIn, classLabels, C, toler, kTup):
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = np.shape(dataMatIn)[0]
        self.alphas = np.mat(np.zeros((self.m, 1)))
        self.b = 0
        self.eCache = np.mat(np.zeros((self.m, 2)))  # error cache
        self.K = np.mat(np.zeros((self.m, self.m)))
        # kTup 是包含核函数信息的元组
        for i in range(self.m):
            self.K[:, i] = kernelTrans(self.X, self.X[i, :], kTup)


def calcEk(oS, k):
    """计算分类误差"""
    fXk = float(np.multiply(oS.alphas, oS.labelMat).T * oS.K[:, k]) + oS.b
    Ek = fXk - float(oS.labelMat[k])

    return Ek


def selectJ(i, oS, Ei):
    """内循环的启发式算法"""
    maxK = -1
    maxDeltaE = 0
    Ej = 0
    # 误差缓存的第一个是是否有效的标志位, 第二位表示实际的误差值
    oS.eCache[i] = [1, Ei]
    validEcacheList = np.nonzero(oS.eCache[:, 0].A)[0]  # 构建了一个非零列表

    if len(validEcacheList) > 1:
        # 误差缓存列表不为空， 遍历每个元素, 取最大误差 (步长）
        for k in validEcacheList:
            if k == i:
                continue
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            if deltaE > maxDeltaE:
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek
        return maxK, Ej
    else:
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej


def updateEk(oS, k):
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1, Ek]


def innerL(i, oS):
    """内循环"""
    Ei = calcEk(oS, i)  # 外循环选择了 ai, 计算误差

    if ((oS.labelMat[i] * Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or \
            ((oS.labelMat[i] * Ei > oS.tol) and (oS.alphas[i] > 0)):
        j, Ej = selectJ(i, oS, Ei)  # 启发式方法搜索 aj, 而非随机选择

        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy()

        if oS.labelMat[i] != oS.labelMat[j]:
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L == H:
            print("H==J")
            return 0

        # aj 的最优修改量
        eta = 2.0 * oS.K[i, j] - oS.K[i, i] - oS.K[j, j]
        if eta >= 0:
            print("eta >= 0")
            return 0
        # 根据修改量修改 aj
        oS.alphas[j] -= oS.labelMat[j][0] * (Ei - Ej) / eta
        oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)
        updateEk(oS, j)  # 更新误差

        # aj 只有轻微修改
        if abs(oS.alphas[j] - alphaJold) < 0.00001:
            print("j not moving enough")
            return 0
        oS.alphas[i] += oS.labelMat[j] * oS.labelMat[i] * (alphaJold - oS.alphas[j])  # 对 ai 进行修改, 修改量同 aj, 但方向相反

        updateEk(oS, i)  # 同样地更新误差, 此处 ai 已经做过修改了

        # 设置常数项 b, 实践中应该是取平均值吧
        b1 = oS.b - Ei - \
             oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i, i] - \
             oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.K[j, j]
        b2 = oS.b - Ej - \
             oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i, j] - \
             oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.K[j, j]
        if 0 < oS.alphas[i] < oS.C:
            oS.b = b1
        elif 0 < oS.alphas[j] < oS.C:
            oS.b = b2
        else:
            oS.b = (b1 + b2) / 2.0
        return 1
    else:
        return 0


def smoP(dataMatIn, classLabels, C, toler, maxIter, kTup=("lin", 0)):
    """完整版 Platt SMO 算法"""
    oS = optStruct(np.mat(dataMatIn), np.mat(classLabels).transpose(), C, toler, kTup)
    ite = 0
    entireSet = True
    alphaPairsChanged = 0

    # 更多的退出条件: 迭代次数超过指定最大值, 或者遍历整个集合都未对任意 alpha-pair 进行修改
    # 此处的迭代定义为一次循环过程: 在优化过程中存在波动就会停止
    while ite < maxIter and (alphaPairsChanged > 0 or entireSet):
        alphaPairsChanged = 0

        if entireSet:  # 遍历所有
            for i in range(oS.m):
                # 调用 innerL, 有任意一对 alpha 值发生改变, 就返回了 1. 结束本次循环
                alphaPairsChanged += innerL(i, oS)
                print("Full set, iter: %d, i: %d, pairs changed: %d" % (ite, i, alphaPairsChanged))
            ite += 1
        else:  # 遍历非边界值
            nonBoundIds = np.nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIds:
                alphaPairsChanged += innerL(i, oS)
                print("Non-bound, iter: %d, i: %d, pairs changed: %d" % (ite, i, alphaPairsChanged))
            ite += 1
        if entireSet:
            entireSet = False
        elif alphaPairsChanged == 0:  # 执行内循环, alpha-pair 没有改变, 即没有找到合适的, 重新遍历
            entireSet = True
        print("Iteration number: %d" % ite)
    return oS.b, oS.alphas


def calcWs(alphas, dataArr, classLabels):
    X = np.mat(dataArr)
    labelMat = np.mat(classLabels).transpose()
    m, n = X.shape
    w = np.zeros((n, 1))
    for i in range(m):
        w += np.multiply(alphas[i] * labelMat[i], X[i, :].T)
    return w


def img2vector(filename):
    returnVect = np.zeros((1, 1024))
    with open(filename) as f:
        for i in range(32):
            lineStr = f.readline()
            for j in range(32):
                returnVect[0, 32 * i + j] = int(lineStr[j])
        return returnVect


def loadImages(dirName):
    from os import listdir

    hwLabels = []
    trainingFileList = listdir(dirName)
    m = len(trainingFileList)
    trainingMat = np.zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split(".")[0]
        classNumStr = int(fileStr.split("_")[0])
        if classNumStr == 9:
            hwLabels.append(-1)
        else:
            hwLabels.append(1)
        trainingMat[i, :] = img2vector("%s/%s" % (dirName, fileNameStr))

    return trainingMat, hwLabels


def testDigits(kTup=("rbf", 10)):
    dataArr, labelArr = loadImages("trainingDigits")
    b, alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, kTup)
    dataMat = np.mat(dataArr)
    labelMat = np.mat(labelArr).transpose()
    svInd = np.nonzero(alphas.A > 0)[0]
    sVs = dataMat[svInd]
    labelSV = labelMat[svInd]
    print("There are %d Support Vectors" % sVs.shape[0])

    m, n = dataMat.shape
    errorCount = 0
    for i in range(m):
        # 给出了如何利用核函数进行分类:
        # 利用 kernelTrans 得到转换后的数据
        # 再利用 alpha 及类别标签值求积
        kernelEval = kernelTrans(sVs, dataMat[i, :], kTup)
        predict = kernelEval.T * np.multiply(labelSV, alphas[svInd]) + b
        if np.sign(predict) != np.sign(labelArr[i]):
            errorCount += 1
    print("The training error rate is: %f" % (float(errorCount) / m))

    dataArr, labelArr = loadImages("testDigits")
    errorCount = 0
    dataMat = np.mat(dataArr)
    labelMat = np.mat(labelArr).transpose()
    m, n = dataMat.shape
    for i in range(m):
        kernelEval = kernelTrans(sVs, dataMat[i, :], kTup)
        predict = kernelEval.T * np.multiply(labelSV, alphas[svInd]) + b
        if np.sign(predict) != np.sign(labelArr[i]):
            errorCount += 1
    print("The test error rate is: %f" % (float(errorCount) / m))


if __name__ == '__main__':
    testDigits(("rbf", 20))
