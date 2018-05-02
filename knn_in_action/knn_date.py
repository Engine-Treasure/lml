# -*- coding: utf-8 -*-

import numpy as np
import operator


def file2matrix(filename):
    with open(filename) as f:
        arrayOLines = f.readlines()
        numberOfLines = len(arrayOLines)
        returnMat = np.zeros((numberOfLines, 3))
        classLabelVector = []

        index = 0
        for l in arrayOLines:
            l = l.strip()
            listFromLine = l.split("\t")
            returnMat[index, :] = listFromLine[0:3]
            classLabelVector.append(int(listFromLine[-1]))
            index += 1
        return returnMat, classLabelVector


def autoNorm(dataSet):
    minVals = dataSet.min(0)  # 0 表示列最小值, 返回的是各列最小值的向量
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    m = dataSet.shape[0]
    # 我觉得示例代码的做法有点繁琐了
    normDataSet = dataSet - np.tile(minVals, (m, 1))
    normDataSet = normDataSet / np.tile(ranges, (m, 1))  # 除法对于 NumPy 而言, 是特征值相除
    return normDataSet, ranges, minVals


def classify0(inX, dataSet, labels, k):
    """

    :param inX: 输入向量, 待分类的实例
    :param dataSet: 数据集
    :param labels: 标签向量
    :param k:
    :return:
    """
    dataSetSize = dataSet.shape[0]

    # 距离计算, 针对空间中所有点计算距离, 效率低下可见一斑
    # np.tile - 重复 inX 以构造矩阵, 得到的 diffMat 就是差的矩阵 (element-wise)
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2  # element-wise, 计算平方差
    sqDistances = sqDiffMat.sum(axis=1)  # 求和
    distances = sqDistances ** 0.5  # 开根号

    # 升序排序
    sortedDistIndicies = distances.argsort()

    # 统计近邻的标签, 并排序
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sorteClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)

    return sorteClassCount[0][0]


def datingClassTest():
    hoRatio = 0.10  # 测试集占数据集大小的比值
    datingDataMat, datingLabels = file2matrix("datingTestSet2.txt")
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0

    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :],
                                     datingLabels[numTestVecs:m], 3)
        print("The classifier caome back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
        if classifierResult != datingLabels[i]:
            errorCount += 1.0
    print("The total error rate is : %f" % (errorCount / float(numTestVecs)))


def classifyPerson():
    resultList = ["not at all", "is small doses", "in large doses"]
    percentTats = float(input("Percentage of time spent playing vedio games?"))
    ffMiles = float(input("frequent flier miles eared per year?"))
    iceCream = float(input("Liters of ice cream consumed per year?"))

    datingDataMat, datingLabels = file2matrix("datingTestSet2.txt")
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = np.array([ffMiles, percentTats, iceCream])

    classifierResult = classify0((inArr - minVals) / ranges, normMat, datingLabels, 3)
    print("You will probably like this person:", resultList[classifierResult - 1])







if __name__ == '__main__':
    classifyPerson()
