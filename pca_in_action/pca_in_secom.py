# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


def loadDataSet(fileName, delim="\t"):
    with open(fileName) as f:
        stringArr = [line.strip().split(delim) for line in f.readlines()]
        datArr = [list(map(float, line)) for line in stringArr]
        return np.mat(datArr)


def replaceNanWithMean():
    datMat = loadDataSet("secom.data", " ")
    numFeat = datMat.shape[1]

    for i in range(numFeat):
        # 遍历所有特征, 替换 NaN 为均值
        meanVal = np.mean(datMat[np.nonzero(~np.isnan(datMat[:, i].A))[0], i])
        datMat[np.nonzero(np.isnan(datMat[:, i].A))[0], i] = meanVal
    return datMat


if __name__ == '__main__':
    dataMat = replaceNanWithMean()

    meanVals = np.mean(dataMat, axis=0)
    meanRemoved = dataMat - meanVals  # element-wise

    covMat = np.cov(meanRemoved, rowvar=0)
    # 会发现很多特征值都是 0, 意味着它们都是其他特征的副本, 可以通过其他特征来表示, 而本身并没有提供额外的信息
    eigVals, eigVecs = np.linalg.eig(np.mat(covMat))

    eigValInd = np.argsort(eigVals)  # 对特征值从小到大排序, 得到索引
    eigValInd = eigValInd[:: -1]  # 降序, 从大到小, 并筛选 k 个特征值
    eigVals = eigVals[eigValInd]

    print(eigVals)
    plt.plot(np.arange(eigVals.shape[0]), eigVals, marker="^")
    plt.show()
