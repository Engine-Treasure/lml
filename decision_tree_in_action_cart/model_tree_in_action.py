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


def linearSolve(dataSet):
    m, n = dataSet.shape

    # 将数据集格式化为目标变量 Y 和自变量 X, 用于执行简单的线性回归
    X = np.mat(np.ones((m, n)))
    X[:, 1:n] = dataSet[:, 0: n-1]
    Y = dataSet[:, -1]

    xTx = X.T * X
    # np.linalg.det 计算行列式
    if np.linalg.det(xTx) == 0.0:
        raise NameError("This matrix is singular, can not do inverse, \n \
                        try increasing the second value of ops")
    # .I 求矩阵的逆
    # 这是正规方程呀
    ws = xTx.I * (X.T * Y)
    return ws, X, Y


def modelLeaf(dataSet):
    ws, X, Y = linearSolve(dataSet)
    # ws 是权值矩阵
    return ws


def modelErr(dataSet):
    ws, X, Y = linearSolve(dataSet)
    # yHat 是估计值
    yHat = X * ws
    return sum(np.power(Y - yHat, 2))


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
    for featIndex in range(n - 1):
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

    mat0 = dataSet[np.nonzero(dataSet[:, feature] > value), :][0]
    mat1 = dataSet[np.nonzero(dataSet[:, feature] <= value), :][0]
    return mat0, mat1


def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    """

    :param dataSet:  训练数据集
    :param leafType: 建立叶节点的函数
    :param errType: 误差计算函数
    :param ops: 包含树构建所需参数的元组
    :return:
    """
    # 首次尝试进行最优切分, 得到特征索引及最优切分的特征值
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


def isTree(obj):
    return type(obj).__name__ == "dict"


def getMean(tree):
    """遍历树, 直到叶节点为止, 找到叶节点, 计算它们的平均值"""
    if isTree(tree["right"]):
        tree["right"] = getMean(tree["right"])
    if isTree(tree["left"]):
        tree["left"] = getMean(tree["left"])

    return (tree["left"] + tree["right"]) / 2.0


def prune(tree, testData):
    # 没有测试数据, 对树进行塌陷处理 (即返回树的平均值)
    if testData.shape[0] == 0:
        return getMean(tree)

    if isTree(tree["left"]) or isTree(tree["right"]):
        # 当前节点存在子树的, 则对当前数据集进行切分, 按当前节点的索引和值进行切分
        lSet, rSet = binSplitDataSet(testData, tree["spInd"], tree["spVal"])

    if isTree(tree["left"]):
        # 上一步切分所得左子集, 用于左子树的再剪枝
        tree["left"] = prune(tree["left"], lSet)
    if isTree(tree["right"]):
        tree["right"] = prune(tree["right"], rSet)

    if not isTree(tree["left"]) and not isTree(tree["right"]):
        # 递归终止条件, 两边都是叶节点
        lSet, rSet = binSplitDataSet(testData, tree["spInd"], tree["spVal"])

        # 误差以平方差和进行计算, tree["left"] 是左叶节点,
        # lSet[:, -1] - tree["left"] 是划分的左子集的每个元素都减去左叶节点的值, 再求平方, 再求和
        errorNoMerge = sum(np.power(lSet[:, -1] - tree["left"], 2)) + \
                       sum(np.power(rSet[:, -1] - tree["right"], 2))

        treeMean = (tree["left"] + tree["right"]) / 2.0
        errorMerge = sum(np.power(testData[:, -1] - treeMean, 2))

        if errorMerge < errorNoMerge:
            print("Merging")
            return treeMean
        else:
            return tree
    else:
        return tree


if __name__ == '__main__':
    myData2 = loadDataSet("exp2.txt")
    myMat2 = np.mat(myData2)
    regTree2 = createTree(myMat2, modelLeaf, modelErr, (1, 10))
    print("Before pruning:", regTree2)

    plt.scatter(myMat2[:, 0].A1, myMat2[:, 1].A1)
    plt.show()
