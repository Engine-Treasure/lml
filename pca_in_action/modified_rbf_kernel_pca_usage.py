# -*- coding: utf-8 -*-


import matplotlib.pyplot as plt
import numpy as np
from scipy import exp
from scipy.linalg import eigh
from scipy.spatial.distance import pdist, squareform
from sklearn.datasets import make_moons
from sklearn.decomposition import PCA


def rbf_kernel_pca(X, gamma, n_components):
    """
    :param X:
    :param gamma: 径向基核的调节参数
    :param n_components: 返回的主成分数量
    :return:
        X_pc: 映射后的数据集
        lambdas: 核矩阵的特征值
    """

    # 对 MxN 维数据计算欧式距离, pairwise
    sq_dists = pdist(X, "sqeuclidean")

    # 将点间距离转换成方阵
    mat_sq_dists = squareform(sq_dists)

    # 计算系统的核矩阵
    K = exp(-gamma * mat_sq_dists)

    # 中心化核矩阵
    N = K.shape[0]
    one_n = np.ones((N, N)) / N
    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)

    # 获得特征值与特征向量, 按序排列好了
    eigvals, eigvecs = eigh(K)


    # 选择 top k 特征向量
    alphas = np.column_stack((eigvecs[:, -i] for i in range(1, n_components+1)))

    # 选择相应的特征值
    lambdas = [eigvals[-i] for i in range(1, n_components+1)]

    return alphas, lambdas


def project_x(x_new, X, gamma, alphas, lambdas):
    """映射用到了原始数据集 X"""
    pair_dist = np.array([np.sum((x_new - row)**2) for row in X])
    k = np.exp(-gamma * pair_dist)
    return k.dot(alphas / lambdas)

X, y = make_moons(n_samples=100, random_state=123)
alphas, lambdas = rbf_kernel_pca(X, gamma=15, n_components=1)

x_new = X[27]
x_proj = alphas[27]  # 原始映射

x_reproj = project_x(x_new, X, gamma=15, alphas=alphas, lambdas=lambdas)

print("Origin projection:", x_proj)
print("Reprojection:", x_reproj)

plt.scatter(alphas[y == 0, 0], np.zeros((50, 1)), color="red", marker="^", alpha=0.5)
plt.scatter(alphas[y == 1, 0], np.zeros((50, 1)), color="blue", marker="o", alpha=0.5)
plt.scatter(x_proj, 0, color="black", marker="^", label="origin projection of point X[27]" )
plt.scatter(x_reproj, 0, color="green", marker="x", s=500, label="Remapped point X[27]" )

plt.legend(scatterpoints=1)
plt.show()
