# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


def loadDataSet(fileName):
    dataMat = []

    with open(fileName) as f:
        for line in f.readlines():
            curLine = line.strip().split("\t")
            fltLine = list(map(float, curLine))
            dataMat.append(fltLine)

        return dataMat



def regLeaf(dataSet):
    """回归树的叶节点, 返回类的平均值"""
    return np.mean(dataSet[:, -1])


def regErr(dataSet):
    """回归树的误差计算: 平方误差"""
    # np.var - 计算均方差
    # 乘以数据集样本中的个数, 得到总方差
    return np.var(dataSet[:, -1]) * dataSet.shape[0]



def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    """
    该函数的目的是找到数据的最佳二元切分方式, 若找不到一个好的二元切分, 返回 None, 并调用 regLeaf 来产生叶节点
    该函数涉及到的提前终止条件, 实际是预剪枝的手段
    :return:
    """
    # 用于控制函数停止时机
    tolS = ops[0]  # 容许的误差下降值
    tolN = ops[1]  # 切分的最少样本数

    # 所有值相等, 返回
    if len(set(dataSet[:, -1].T.tolist()[0])) == 1:  # 剩余特征值的数目
        return None, leafType(dataSet)

    m, n = dataSet.shape  # 当前数据集的大小
    S = errType(dataSet)  # 整个数据集的误差, 误差用于与新切分误差进行对比, 检查新切分能否降低误差

    bestS = np.inf  # 最大误差, 初始为无穷大; 最佳切分是使得切分后的误差达到最低的切分
    bestIndex = 0
    bestValue = 0
    for featIndex in range(n-1):
        # 遍历所有特征
        for splitVal in set(dataSet[:, featIndex].A1):
            # 遍历所有可切分的值
            # 先查看切分后的子集大小
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            if mat0.shape[0] < tolN or mat1.shape[0] < tolN:
                continue

            # 在对比切分后的误差
            newS = errType(mat0) + errType(mat1)
            if newS < bestS:
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS

    # 误差减小不大, 不必进行切分, 而直接返回叶节点
    if (S - bestS) < tolS:
        return None, leafType(dataSet)

    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)

    # 切分出的数据集很小, 小于用户定义的大小, 也直接返回叶节点
    if mat0.shape[0] < tolN or mat1.shape[0] < tolN:
        return None, leafType(dataSet)

    return bestIndex, bestValue




def binSplitDataSet(dataSet, feature, value):
    """对给定特征, 按特征值二分数据集"""

    # print(np.nonzero(dataSet[dataSet[:, feature] > value]))
    # mat0 = dataSet[:, dataSet[:, feature] > value]
    mat0 = dataSet[np.nonzero(dataSet[:, feature] > value), :][0]
    mat1 = dataSet[np.nonzero(dataSet[:, feature] <= value), :][0]
    return mat0, mat1



def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    """

    :param dataSet:
    :param leafType: 建立叶节点的函数
    :param errType: 误差计算函数
    :param ops: 包含树构建所需其他参数的元组
    :return:
    """
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)
    if feat == None:
        # 满足递归条件, 返回叶节点值. 若构建的是回归树, val 是一个常数; 若构建的是模型树, val 是一个线性方程
        return val

    retTree = {}
    retTree["spInd"] = feat
    retTree["spVal"] = val

    # 不满足停止条件, 继续递归切分
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    retTree["left"] = createTree(lSet, leafType, errType, ops)
    retTree["right"] = createTree(rSet, leafType, errType, ops)

    return retTree


if __name__ == '__main__':
    myData = loadDataSet("ex00.txt")
    myMat = np.mat(myData)
    regTree = createTree(myMat)
    print(regTree)

    plt.scatter(myMat[:, 0].A1, myMat[:, 1].A1)
    plt.show()

    myData1 = loadDataSet("ex0.txt")
    myMat1 = np.mat(myData1)
    regTree1 = createTree(myMat1)
    print(regTree1)


    plt.scatter(myMat1[:, 1].A1, myMat1[:, 2].A1)
    plt.show()


    myData2 = loadDataSet("ex2.txt")
    myMat2 = np.mat(myData2)
    regTree2 = createTree(myMat2)
    print(regTree2)


    plt.scatter(myMat2[:, 0].A1, myMat2[:, 1].A1)
    plt.show()
