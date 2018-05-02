# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
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
                print("Split: dim %d, thresh %.2f, thresh inequal: %s, the weighted error is %.3f" %
                      (i, threshVal, inequal, weightedError))
                if weightedError < minError:
                    minError = weightedError
                    bestClassEst = predictedVals.copy()
                    bestStump["dim"] = i
                    bestStump["thresh"] = threshVal
                    bestStump["ineq"] = inequal
    return bestStump, minError, bestClassEst


if __name__ == '__main__':
    mat, labels = loadSimpData()
    D = np.mat(np.ones((5, 1)) / 5)
    result = buildStump(mat, labels, D)
    print(result)
