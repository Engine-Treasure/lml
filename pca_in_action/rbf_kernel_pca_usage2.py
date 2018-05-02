# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from scipy import exp
from scipy.linalg import eigh
from scipy.spatial.distance import pdist, squareform
from sklearn.datasets import make_circles
from sklearn.decomposition import PCA


def rbf_kernel_pca(X, gamma, n_components):
    """
    :param X:
    :param gamma: 径向基核的调节参数
    :param n_components: 返回的主成分数量
    :return: 映射后的数据集
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
    X_pc = np.column_stack((eigvecs[:, -i] for i in range(1, n_components + 1)))

    return X_pc


X, y = make_circles(n_samples=1000, random_state=123, noise=0.1, factor=0.2)
plt.scatter(X[y == 0, 0], X[y == 0, 1], color="red", marker="^", alpha=0.5)
plt.scatter(X[y == 1, 0], X[y == 1, 1], color="blue", marker="o", alpha=0.5)

plt.show()

# standard pca
sklearn_pca = PCA(n_components=2)
X_spca = sklearn_pca.fit_transform(X)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7, 3))
ax[0].scatter(X_spca[y == 0, 0], X_spca[y == 0, 1], color="red", marker="^", alpha=0.5)
ax[0].scatter(X_spca[y == 1, 0], X_spca[y == 1, 1], color="blue", marker="o", alpha=0.5)
ax[1].scatter(X_spca[y == 0, 0], np.zeros((500, 1)) + 0.02, color="red", marker="^", alpha=0.5)
ax[1].scatter(X_spca[y == 1, 0], np.zeros((500, 1)) - 0.02, color="blue", marker="o", alpha=0.5)
ax[0].set_xlabel("PC1")
ax[0].set_ylabel("PC2")
ax[1].set_ylim([-1, 1])
ax[1].set_yticks([])
ax[1].set_xlabel("PC1")
plt.show()

# rbf kernel pca
X_kpca = rbf_kernel_pca(X, gamma=15, n_components=2)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7, 3))
ax[0].scatter(X_kpca[y == 0, 0], X_kpca[y == 0, 1], color="red", marker="^", alpha=0.5)
ax[0].scatter(X_kpca[y == 1, 0], X_kpca[y == 1, 1], color="blue", marker="o", alpha=0.5)
ax[1].scatter(X_kpca[y == 0, 0], np.zeros((500, 1)) + 0.02, color="red", marker="^", alpha=0.5)
ax[1].scatter(X_kpca[y == 1, 0], np.zeros((500, 1)) - 0.02, color="blue", marker="o", alpha=0.5)
ax[0].set_xlabel("PC1")
ax[0].set_ylabel("PC2")
ax[1].set_ylim([-1, 1])
ax[1].set_yticks([])
ax[1].set_xlabel("PC1")
plt.show()
