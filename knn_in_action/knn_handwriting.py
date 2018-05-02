# -*- coding: utf-8 -*-

from os import listdir

import numpy as np
import operator

def img2vector(filename):
    returnVect = np.zeros((1, 1024))
    with open(filename) as f:
        for i in range(32):
            lineStr = f.readline()
            for j in range(32):
                returnVect[0, 32*i+j] = int(lineStr[j])
        return returnVect


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
def handwritingClasstTest():
    hwLabels = []

    trainingFileList = listdir("trainingDigits")
    m = len(trainingFileList)
    trainingMat = np.zeros((m, 1024))

    for i in range(m):
        # 没有训练过程, 仅录入数据
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split(".")[0]
        classNumberStr = int(fileStr.split("_")[0])
        hwLabels.append(classNumberStr)
        trainingMat[i, :] = img2vector("trainingDigits/%s" % fileNameStr)

    testFileList = listdir("testDigits")
    errorCount = 0.0
    mTest = len(testFileList)

    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split(".")[0]
        classNumberStr = int(fileStr.split("_")[0])

        vectorUnderTest = img2vector("testDigits/%s" % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)

        print("The classifier came back with: %d, the real  answer is: %d" % (classifierResult, classNumberStr))
        if classifierResult != classNumberStr:
            errorCount += 1.0

    print("The total error number is : %d" % errorCount)
    print("The total error rate is : %f" % (errorCount / float(mTest)))


if __name__ == '__main__':
    handwritingClasstTest()
