# -*- coding: utf-8 -*-


from scipy.spatial.distance import pdist, squareform
from scipy import exp
from scipy.linalg import eigh
import numpy as np


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
    X_pc = np.column_stack((eigvecs[:, -i] for i in range(1, n_components+1)))

    return X_pc
