# -*- coding: utf-8 -*-

import numpy as np
from math import log
import operator


def majorityCnt(classList):
    """多数投票表决"""

    classCount = {}  # 类标签: 统计
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)

    return sortedClassCount[0][0]


def createDataSet():
    dataSet = [
        [1, 1, "yes"],
        [1, 1, "yes"],
        [1, 0, "no"],
        [0, 1, "no"],
        [0, 1, "no"],
    ]
    labels = ["no surfacing", "flippers"]
    return dataSet, labels


def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt



def splitDataSet(dataSet, axis, value):
    """
    :param dataSet: 待分类的数据集
    :param axis: 划分数据集的特征
    :param value: 返回的特征值
    :return:
    """
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            # 抽取
            # 以下操作将一个列表, 划分成了两个. 此时, axis 位置的元素将被丢弃
            reducedFeatVec = featVec[: axis]
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet



def chooseBestFeatureToSplit(dataSet):
    """
    该函数实现了选取特征, 划分数据集, 计算得出最好的划分数据集的特征
    """
    numFeatures = len(dataSet[0]) - 1  # 特征数, 减一是尾元素是类别标签
    baseEntropy = calcShannonEnt(dataSet)  # 数据集的原始香农熵, 即 H(D)
    bestInfoGain = 0.0
    bestFeature = -1

    for i in range(numFeatures):
        # 遍历数据集中的所有特征

        # 创建唯一的分类标签列表
        featList = [sample[i] for sample in dataSet]  # 只取了数据集中所有第 i 个特征值
        uniqueVals = set(featList)

        newEntropy = 0.0
        for value in uniqueVals:
            # 遍历当前特征中的所有唯一属性值

            subDataSet = splitDataSet(dataSet, i, value)  # 对每个唯一属性值划分一次数据集
            prob = len(subDataSet) / float(len(dataSet))  # 计算数据集的新熵
            newEntropy += prob * calcShannonEnt(subDataSet)  # 对熵求和

        # 信息增益是熵的减少或者数据无序度的减少
        infoGain = baseEntropy - newEntropy  # 熵增益就是 H(D) - H(D;A)
        if infoGain > bestInfoGain:
            # 比较所有特征中的信息增益, 返回最好的特征划分的索引值
            bestInfoGain = infoGain
            bestFeature = i

    return bestFeature



def createTree(dataSet, labels):
    classList = [sample[-1] for sample in dataSet]  # 类的列表

    # 递归的第一个停止条件
    # 类别完全相同的情况, 停止继续划分
    if classList.count(classList[0]) == len(classList):
        return classList[0]

    # 遍历完所有特征, 仍无法划分数据集
    # 遍历完所有特征时 进行多数表决
    if len(dataSet[0]) == 1:  # =1, 表示只剩下类标签了, 即所有的特征都被使用过了
        return majorityCnt(classList)

    bestFeat = chooseBestFeatureToSplit(dataSet)  # 就当前数据集, 选取了分类效果最好的特征
    bestFeatLabel = labels[bestFeat]  # 选择对应的标签名

    myTree = {bestFeatLabel: {}}
    del(labels[bestFeat])  # 使用过的特征不再使用, 因此, 删除对应的标签名

    # 得到特征下, 所有属性值的列表
    featValues = [sample[bestFeat] for sample in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        # 遍历当前选择特征包含的所有属性值, 在每个数据集划分上递归调用函数
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)

    return myTree



def classify(inputTree, featLabels, testVec):
    firstStr = list(inputTree.keys)[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)  # 定位特征在特征标签列表中的位置

    for key in secondDict.keys():
        # 遍历树, 比较 testVec 中的值与树节点树
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == "dict":
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                # 达到叶节点, 返回类别
                classLabel = secondDict[key]
    return classLabel



def storeTree(inputTree, filename):
    import pickle
    with open(filename, "w") as f:
        pickle.dump(inputTree, f)


def grabTree(filename):
    import pickle
    with open(filename) as f:
        return pickle.load(f)


if __name__ == '__main__':
    myData , labels = createDataSet()
    print(calcShannonEnt(myData))
    print(splitDataSet(myData, 2, "yes"))

    print(chooseBestFeatureToSplit(myData))
    myTree = createTree(myData, labels)
    print(myTree)
