import numpy as np


def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))


def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    """梯度上升法， 求 weights"""
    m, n = np.shape(dataMatrix)
    weights = np.ones(n)
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 1 / (1.0 + j + i) + 0.01  # 动态修正学习率
            randIndex = int(np.random.choice(dataIndex))  # 随机选择一条记录
            h = sigmoid(sum(dataMatrix[randIndex] * weights))
            error = classLabels[randIndex] - h
            weights += alpha * error * dataMatrix[randIndex]
            dataIndex.remove(randIndex)  # 用完就丢, 这一次 j 迭代不再使用

    return weights


def classifyVector(inX, weights):
    prob = sigmoid(sum(inX * weights))
    return 1.0 if prob > 0.5 else 0.0


def colicTest():
    trainningSet = []
    trainningLabels = []

    # with open("horse-colic.data") as frTrain:
    with open("horse-colic.data2") as frTrain:
        for line in frTrain.readlines():
            currLine = line.strip().split()
            # currLine = line.strip().split("\t")
            # lineArr = [0.0 if currLine[i] == "?" else float(currLine[i]) for i in range(27)]  # 行属性
            lineArr = [float(currLine[i]) for i in range(21)]  # 行属性

            trainningSet.append(lineArr)
            # trainningLabels.append(float(currLine[27]))
            trainningLabels.append(float(currLine[21]))

    trainWeights = stocGradAscent1(np.array(trainningSet), trainningLabels, 500)
    errorCount = 0
    numTestVec = 0.0

    # with open("horse-colic.test") as frTest:
    with open("horse-colic.test2") as frTest:
        lines = frTest.readlines()
        for line in lines:
            numTestVec += 1.0  # 测试记录数
            currLine = line.strip().split()
            # currLine = line.strip().split("\t")
            # lineArr = [0.0 if currLine[i] == "?" else float(currLine[i]) for i in range(27)]  # 行属性
            lineArr = [float(currLine[i]) for i in range(21)]  # 行属性

            # if int(classifyVector(np.array(lineArr), trainWeights)) != int(currLine[27]):
            if int(classifyVector(np.array(lineArr), trainWeights)) != int(currLine[21]):
                errorCount += 1

    errorRate = float(errorCount) / numTestVec

    print("The error rate of this test is %f" % errorRate)

    return errorRate


def multiTest():
    numTests = 10
    errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print("After %d iterations the average error rate is: %f" % (numTests, errorSum / float(numTests)))


if __name__ == '__main__':
    multiTest()
