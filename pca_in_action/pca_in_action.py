# -*- coding: utf-8 -*-

import numpy as np

def loadDataSet(fileName, delim="\t"):
    with open(fileName) as f:
        stringArr = [line.strip().split(delim) for line in f.readlines()]
        datArr = [list(map(float, line)) for line in stringArr]
        return np.mat(datArr)


def pca(dataMat, topNfeat=9999999):
    # 计算并减去平均值
    meanVals = np.mean(dataMat, axis=0)
    meanRemoved = dataMat - meanVals  # element-wise

    covMat = np.cov(meanRemoved, rowvar=0)
    eigVals, eigVecs = np.linalg.eig(np.mat(covMat))
    eigValInd = np.argsort(eigVals)  # 对特征值从小到大排序, 得到索引
    eigValInd = eigValInd[: -(topNfeat + 1): -1]  # 降序, 从大到小, 并筛选 k 个特征值
    redEigVecs = eigVecs[:, eigValInd]  # 获得 top k 特征值对应的特征向量, 直接构成了映射矩阵
    lowDDataMat = meanRemoved * redEigVecs  # 转换后的低维矩阵
    reconMat = (lowDDataMat * redEigVecs.T) + meanVals  # 重构原数据?
    return lowDDataMat, reconMat


if __name__ == '__main__':
    dataMat = loadDataSet("testSet.txt")
    lowDMat, reconMat = pca(dataMat, 1)

    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dataMat[:, 0].flatten().A[0], dataMat[:, 1].flatten().A[0], marker="^", s=90)
    ax.scatter(reconMat[:, 0].flatten().A[0], reconMat[:, 1].flatten().A[0], marker="o", s=50, color="red")
    plt.show()

