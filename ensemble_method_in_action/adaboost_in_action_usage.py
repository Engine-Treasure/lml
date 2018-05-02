# -*- coding: utf-8 -*-

import numpy as np


def loadSimpData():
    datMat = np.matrix([
        [1.0, 2.1],
        [2.0, 1.1],
        [1.3, 1.0],
        [1.0, 1.0],
        [2.0, 1.0]
    ])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat, classLabels


def loadDataSet(fileName):
    with open(fileName) as f:
        numFeat = len(f.readline().split("\t"))
    dataMat = []
    labelMat = []

    with open(fileName) as f:
        for line in f.readlines():
            lineArr = []
            curLine = line.strip().split("\t")

            for i in range(numFeat - 1):
                lineArr.append(float(curLine[i]))

            dataMat.append(lineArr)
            labelMat.append(float(curLine[-1]))
        return dataMat, labelMat


def loadDataSet(fileName):
    with open(fileName) as f:
        numFeat = len(f.readline().split("\t"))
    dataMat = []
    labelMat = []

    with open(fileName) as f:
        for line in f.readlines():
            lineArr = []
            curLine = line.strip().split("\t")

            for i in range(numFeat - 1): lineArr.append(float(curLine[i]))

            dataMat.append(lineArr)
            labelMat.append(float(curLine[-1]))
        return dataMat, labelMat


def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    """通过阈值比较对数据进行分类"""
    retArray = np.ones((dataMatrix.shape[0], 1))
    if threshIneq == "lt":
        # 通过布尔过滤来实现
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:, dimen] > threshVal] = 1.0
    return retArray


def buildStump(dataArr, classLabels, D):
    """
    遍历 stumpClassify 函数的所有可能输入值, 并找到数据集上最佳的单层决策树
    '最佳' 是基于数据的权重向量 D 来定义的
    """

    dataMatrix = np.mat(dataArr)
    labelMat = np.mat(classLabels).T
    m, n = dataMatrix.shape
    numSteps = 10.0  # 用于在特征的所有可能值上进行遍历
    bestStump = {}  # 存储决策树桩的相关信息
    bestClassEst = np.mat(np.zeros((m, 1)))
    minError = np.inf  # 最小错误率

    for i in range(n):
        # 在数据集的所有特征上遍历
        # 通过计算最小值与最大值来了解需要的步长
        rangeMin = dataMatrix[:, i].min()
        rangeMax = dataMatrix[:, i].max()
        stepSize = (rangeMax - rangeMin) / numSteps  # 步长是一个浮点数

        for j in range(-1, int(numSteps) + 1):
            # 在特征值上进行遍历, 允许阈值在取值范围之外

            for inequal in ["lt", "gt"]:
                # 在大于和小于之间切换不等式

                threshVal = (rangeMin + float(j) * stepSize)  # 阈值
                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)  # i 是特征标签
                errArr = np.mat(np.ones((m, 1)))
                errArr[predictedVals == labelMat] = 0  # 预测正确, 不计算错误值
                weightedError = D.T * errArr  # 计算加权错误率
                # print("Split: dim %d, thresh %.2f, thresh inequal: %s, the weighted error is %.3f" %
                #       (i, threshVal, inequal, weightedError))
                if weightedError < minError:
                    minError = weightedError
                    bestClassEst = predictedVals.copy()
                    bestStump["dim"] = i
                    bestStump["thresh"] = threshVal
                    bestStump["ineq"] = inequal
    return bestStump, minError, bestClassEst


def adaBoostTrainDS(dataArr, classLabels, numIt=40):
    """
    AdaBoost with Decision Stump
    :param dataArr:
    :param classLabels:
    :param numIt: 最大迭代次数
    :return:
    """

    weakClassArr = []  # 弱分类器的数组
    m = np.shape(dataArr)[0]  # 数据点的数目
    # 存储每个数据点的权重, 用于之后更新.
    # D 为概率分布向量, 因此元素之和为 1.0
    D = np.mat(np.ones((m, 1)) / m)
    aggClassEst = np.mat(np.zeros((m, 1)))  # 记录每个数据点的类别估计累积值

    for i in range(numIt):
        # 1. 建立单层决策树
        # 输入权值向量 D, 返回利用 D 得到的具有最小错误率的单层决策树, 最小错误率, 估计的类别向量
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)
        # print("D: ", D.T)

        # 2. 计算 alpha 值
        # alpha 决定了单层决策树在最终分类结果的权重
        alpha = float(0.5 * np.log((1.0 - error) / max(error, 1e-16)))  # max 函数是为了确保没有错误时不会发生除零错
        bestStump["alpha"] = alpha  # alpha 值加入 bestStump 中
        weakClassArr.append(bestStump)
        # print("ClassEst:", classEst.T)

        # 3. 更新权重向量 D
        expon = np.multiply(-1 * alpha * np.mat(classLabels).T, classEst)
        D = np.multiply(D, np.exp(expon))
        D = D / D.sum()

        aggClassEst += alpha * classEst  # 错误累加计算, 以确保训练错误率为 0 就提前结束循环
        # print("AggClassEst:", aggClassEst.T)
        aggErrors = np.multiply(np.sign(aggClassEst) != np.mat(classLabels).T, np.ones((m, 1)))
        errorRate = aggErrors.sum() / m
        print("Total Error:", errorRate, "\n")

        if errorRate == 0.0:
            break
    return weakClassArr, aggClassEst


def adaClassify(dataToClass, classifierArr):
    """
    分类函数
    :param dataToClass: 待分类样本
    :param classifierArr: 弱学习者的数组
    :return:
    """
    dataMatrix = np.mat(dataToClass)
    m = dataMatrix.shape[0]
    aggClassEst = np.mat(np.zeros((m, 1)))

    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix, classifierArr[i]["dim"],
                                 classifierArr[i]["thresh"],
                                 classifierArr[i]["ineq"])
        aggClassEst += classifierArr[i]["alpha"] * classEst
        # print(aggClassEst)
    return np.sign(aggClassEst)


def plotROC(predStrengths, classLabels):
    import matplotlib.pyplot as plt

    cur = (1.0, 1.0)  # 绘制光标的位置
    ySum = 0.0  # 用于计算 AUC 的值
    numPosClas = sum(np.array(classLabels) == 1.0)  # 计算正例的数量, 确定了在 y 轴上的步进数目
    yStep = 1 / float(numPosClas)
    xStep = 1 / float(len(classLabels) - numPosClas)
    sortedIndicies = predStrengths.argsort()  # 排序

    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    for index in sortedIndicies.tolist()[0]:
        if classLabels[index] == 1.0:
            # 沿 y 轴方法下降一个步长, 不断降低真正率
            delX = 0
            delY = yStep
        else:
            delX = xStep
            delY = 0
            ySum += cur[1]
        ax.plot([cur[0], cur[0]-delX], [cur[1], cur[1]-delY], c='b')
        cur = (cur[0] - delX, cur[1] - delY)
    ax.plot([0, 1], [0, 1], "b--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curve for AdaBoost Horse Colic Detection System")
    ax.axis([0, 1, 0, 1])

    print("The Area Under the Curve is:", ySum * xStep)

    plt.show()

if __name__ == '__main__':
    dataMat, classLabels = loadDataSet("horseColicTraining2.txt")
    classifierArray, aggClassEst = adaBoostTrainDS(dataMat, classLabels, 10)

    testArr, testLaelsArr = loadDataSet("horseColicTest2.txt")
    prefiction10 = adaClassify(testArr, classifierArray)

    errArr = np.mat(np.ones((67, 1)))
    print(errArr[prefiction10 != np.mat(testLaelsArr).T].sum() / 67)

    plotROC(aggClassEst.T, classLabels)
