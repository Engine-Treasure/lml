# -*- coding: utf-8 -*-

import numpy as np
import operator



def createDataSet():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ["A", "A", "B", "B"]
    return group, labels



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

if __name__ == '__main__':
    dataset = createDataSet()
    print(classify0((0.9, 1.0), dataSet=dataset[0], labels=dataset[1], k=2))
