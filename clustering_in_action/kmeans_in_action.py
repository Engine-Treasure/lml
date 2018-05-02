# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np


def loadDataSet(fileName):
    dataMat = []
    with open(fileName) as f:
        for line in f.readlines():
            curLine = line.strip().split("\t")
            fltLine = list(map(float, curLine))
            dataMat.append(fltLine)
        return dataMat


def distEclud(vecA, vecB):
    """计算欧式距离"""
    return np.emath.sqrt(sum(np.emath.power(vecA - vecB, 2)))


def randCent(dataSet, k):
    """
    为给定数据集构造包含 k 个随机质心的集合
    """
    n = np.shape(dataSet)[1]  # 特征数
    centroids = np.mat(np.zeros((k, n)))

    for j in range(n):
        minJ = min(dataSet[:, j])  # 某一特征的最小值
        rangeJ = float(max(dataSet[:, j]) - minJ)  # 定范围, 确保质心在整个数据集的边界内
        centroids[:, j] = minJ + rangeJ * np.random.rand(k, 1)
    return centroids


def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    m = np.shape(dataSet)[0]  # 样本数
    # 用于存储样本的分配结果, 一列记录集群索引, 一列用于存储误差 (样本到质心的距离)
    clusterAssment = np.mat(np.zeros((m, 2)))
    centroids = createCent(dataSet, k)  # 质心的集合
    clusterChanged = True

    while clusterChanged:
        # 循环: 计算质心-分配-更新质心
        clusterChanged = False

        for i in range(m):
            minDist = np.inf
            minIndex = -1

            for j in range(k):
                # 对每个点遍历质心, 寻找最近的质心
                distJI = distMeas(centroids[j, :].T, dataSet[i, :].T)
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j

            if clusterAssment[i, 0] != minIndex:
                # 索引有所更新, 说明还有的更新
                clusterChanged = True
            clusterAssment[i, :] = minIndex, minDist ** 2
        print(centroids)

        for cent in range(k):
            # 更新质心位置
            # 布尔过滤, 得到某一集群的所有样本
            ptsInClust = dataSet[np.nonzero(clusterAssment[:, 0].A == cent)[0]]
            # 计算均值, 并用均值更新集群中心点
            centroids[cent, :] = np.mean(ptsInClust, axis=0)
    return centroids, clusterAssment


def biKMeans(dataSet, k, distMeas=distEclud):
    m = np.shape(dataSet)[0]  # 样本数
    # 用于存储样本的分配结果, 一列记录集群索引, 一列用于存储误差 (样本到质心的距离)
    clusterAssment = np.mat(np.zeros((m, 2)))

    # 初始集群, 所有的样本都属于该集群
    centroid0 = np.mean(dataSet, axis=0).tolist()[0]
    centList = [centroid0]

    for j in range(m):
        # 遍历所有样本,计算其到质心的距离
        clusterAssment[j, 1] = (distMeas(np.mat(centroid0).T, dataSet[j, :].T) ** 2)[0]

    while len(centList) < k:
        lowestSSE = np.inf

        # 遍历集群列表中的每个集权
        for i in range(len(centList)):
            # 尝试划分集群:
            # 获得某一集群的所有样本点, 对该集群进行划分
            ptsInCurrCluster = dataSet[np.nonzero(clusterAssment[:, 0].A == i)[0], :]
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)

            sseSplit = sum(splitClustAss[:, 1])  # 本次划分产生的 sse
            # 未划分数据的 sse
            sseNoSplit = sum(clusterAssment[np.nonzero(clusterAssment[:, 0].A != i)[0], 1])
            print("sseSplit and notSplit:", sseSplit, sseNoSplit)

            if (sseSplit + sseNoSplit) < lowestSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNoSplit

        # 更新簇的分配结果
        bestClustAss[np.nonzero(bestClustAss[:, 0].A == 1)[0], 0] = len(centList)
        bestClustAss[np.nonzero(bestClustAss[:, 0].A == 0)[0], 0] = bestCentToSplit
        print("The best centroid to split is:", bestCentToSplit)
        print("The len of bestClusterAss is:", len(bestClustAss))

        centList[bestCentToSplit] = bestNewCents[0, :]
        centList.append(bestNewCents[1, :])
        clusterAssment[np.nonzero(clusterAssment[:, 0].A == bestCentToSplit)[0], :] = bestClustAss

    return centList, clusterAssment

if __name__ == '__main__':
    # bikmeans
    X = np.mat(loadDataSet("testSet2.txt"))
    plt.scatter([X[:, 0]], [X[:, 1]])
    myCentroids, clusterAssing = biKMeans(X, 6)
    print(myCentroids)
    plt.scatter([m[:, 0] for m in myCentroids], [m[:, 1] for m in myCentroids], s=50, marker="x", color="Red")
    plt.show()

    exit()

    # kmeans
    X = np.mat(loadDataSet("testSet.txt"))
    plt.scatter([X[:, 0]], [X[:, 1]])
    myCentroids, clusterAssing = kMeans(X, 4)
    plt.scatter([myCentroids[:, 0]], [myCentroids[:, 1]], s=50, marker="x", color="Red")
    plt.show()

