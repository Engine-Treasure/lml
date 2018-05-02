# -*- coding: utf-8 -*-
import random
import numpy as np
import matplotlib.pyplot as plt

from  svm_in_action.util import  plot_svm_illustration

# ********************************
# Completed Version of SMO
# ********************************
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

class optStruct:
    """封装了一些常用数据"""

    def __init__(self, dataMatIn, classLabels, C, toler):
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = np.shape(dataMatIn)[0]
        self.alphas = np.mat(np.zeros((self.m, 1)))
        self.b = 0
        self.eCache = np.mat(np.zeros((self.m, 2)))  # error cache


def calcEk(oS, k):
    """计算分类误差"""
    fXk = float(np.multiply(oS.alphas, oS.labelMat).T * (oS.X * oS.X[k, :].T)) + oS.b
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
        eta = 2.0 * oS.X[i, :] * oS.X[j, :].T - \
              oS.X[i, :] * oS.X[i, :].T - oS.X[j, :] * oS.X[j, :].T
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
             oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.X[i, :] * oS.X[i, :].T - \
             oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.X[i, :] * oS.X[j, :].T
        b2 = oS.b - Ej - \
             oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.X[i, :] * oS.X[j, :].T - \
             oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.X[j, :] * oS.X[j, :].T
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
    oS = optStruct(np.mat(dataMatIn), np.mat(classLabels).transpose(), C, toler)
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

if __name__ == '__main__':
    np.random.seed(0)

    dataArr, labelArr = loadDataSet("testSet.txt")

    # alpha 中非零的对应支持向量
    # b, alphas = smoSimple(dataArr, labelArr, 0.6, 0.001, 40)
    b, alphas = smoP(dataArr, labelArr, 0.6, 0.001, 40)
    ws = calcWs(alphas, dataArr, labelArr)
    print("\n>>>==================================================<<<\n")
    print("ws:", ws)
    print("b:", b)
    print("[alphas > 0]", alphas[alphas > 0])

    print("\n>>>==================================================<<<\n")
    support_vectors = []
    for i in range(100):
        if alphas[i] > 0.0:
            support_vectors.append(dataArr[i])
            print("support vector:", dataArr[i], labelArr[i])
    support_vectors = np.mat(support_vectors)

    plot_svm_illustration(dataArr, labelArr, support_vectors, ws, b)
