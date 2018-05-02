# -*- coding: utf-8 -*-

import random
import numpy as np
import matplotlib.pyplot as plt

from  svm_in_action.util import  plot_svm_illustration


# ********************************
# Simple Version of SMO
# ********************************
def loadDataSet(fileName):
    dataMat = []
    labelMat = []
    with open(fileName) as f:
        for line in f.readlines():
            lineArr = line.strip().split("\t")
            dataMat.append([float(lineArr[0]), float(lineArr[1])])
            labelMat.append(float(lineArr[2]))

    return dataMat, labelMat


def selectJrand(i, m):
    j = i
    while i == j:
        j = int(random.uniform(0, m))
    return j


def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    elif aj < L:
        aj = L
    return aj


def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    dataMatrix = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).transpose()
    b = 0
    m, n = dataMatrix.shape
    alphas = np.mat(np.zeros((m, 1)))

    ite = 0  # 用于存储在没有 alpha 改变的情况下遍历数据集的次数
    while ite < maxIter:
        alphaPairsChanged = 0

        for i in range(m):
            # 预测的类别
            fXi = float(np.multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[i, :].T)) + b  # 疑为 np.matmul
            # 预测结果与真实结果的误差
            Ei = fXi - float(labelMat[i])
            # 判断 alpha　是否可以优化
            # 误差很大, 则可以对数据实例对应的 alpha 进行优化 (违反 kkt 条件严重的)
            if ((labelMat[i] * Ei < -toler) and (alphas[i] < C)) or ((labelMat[i] * Ei > toler) and (alphas[i] > 0)):
                # 随机选择第二个 alpha
                j = selectJrand(i, m)
                fXj = float(np.multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[j, :].T)) + b
                Ej = fXj - float(labelMat[j])

                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()

                # 这一对 if .. else 保证 alpha 在 0 到 C 之间
                if labelMat[i] != labelMat[j]:
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L == H:
                    print("H==J")
                    continue
                # aj 的最优修改量
                eta = 2.0 * dataMatrix[i, :] * dataMatrix[j, :].T - \
                      dataMatrix[i, :] * dataMatrix[i, :].T - dataMatrix[j, :] * dataMatrix[j, :].T
                if eta >= 0:
                    print("eta >= 0")
                    continue
                # 根据修改量修改 aj
                alphas[j] -= labelMat[j][0] * (Ei - Ej) / eta
                alphas[j] = clipAlpha(alphas[j], H, L)

                # aj 只有轻微修改
                if abs(alphas[j] - alphaJold) < 0.00001:
                    print("j not moving enough")
                    continue
                alphas[i] += labelMat[j] * labelMat[i] * (alphaJold - alphas[j])  # 对 ai 进行修改, 修改量同 aj, 但方向相反
                # 设置常数项 b, 实践中应该是取平均值吧
                b1 = b - Ei - \
                     labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i, :] * dataMatrix[i, :].T \
                     - labelMat[j] * (alphas[j] - alphaJold) * dataMatrix[i, :] * dataMatrix[j, :].T
                b2 = b - Ej - \
                     labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i, :] * dataMatrix[j, :].T - \
                     labelMat[j] * (alphas[j] - alphaJold) * dataMatrix[j, :] * dataMatrix[j, :].T
                if 0 < alphas[i] < C:
                    b = b1
                elif 0 < alphas[j] < C:
                    b = b2
                else:
                    b = (b1 + b2) / 2.0
                alphaPairsChanged += 1
                print("iter: %d i: %d, pairs changed %d" % (ite, i, alphaPairsChanged))
        if alphaPairsChanged == 0:
            ite += 1
        else:
            ite = 0
        print("iteration number: %d" % ite)
    return b, alphas


def calcWs(alphas, dataArr, classLabels):
    X = np.mat(dataArr)
    labelMat = np.mat(classLabels).transpose()
    m, n = X.shape
    w = np.zeros((n, 1))
    for i in range(m):
        w += np.multiply(alphas[i] * labelMat[i], X[i, :].T)
    return w


if __name__ == '__main__':
    np.random.seed(0)

    dataArr, labelArr = loadDataSet("testSet.txt")

    # alpha 中非零的对应支持向量
    b, alphas = smoSimple(dataArr, labelArr, 0.6, 0.001, 40)
    ws = calcWs(alphas, dataArr, labelArr)
    print("\n>>>==================================================<<<\n")
    print("ws:", ws)
    print("b:", b)
    print("[alphas > 0]", alphas[alphas > 0])

    print("\n>>>==================================================<<<\n")
    support_vectors = []
    for i in range(100):
        if alphas[i] > 0.0:
            support_vectors.append(dataArr[i])
            print("support vector:", dataArr[i], labelArr[i])
    support_vectors = np.mat(support_vectors)

    plot_svm_illustration(dataArr, labelArr, support_vectors, ws, b)
