# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


df_wine = pd.read_csv("wine.data")
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# 1. 标准化
sc =StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# 2. 构造协方差矩阵
cov_mat = np.cov(X_train_std.T)  # 估计协方差矩阵

# 3. 特征分解, 得到特征值与特征向量
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)  # 对矩阵进行特征分解

tot = sum(eigen_vals)
var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)

plt.bar(range(1, 14), var_exp, alpha=0.5, align="center", label="individual explained variance")
plt.step(range(1, 14), cum_var_exp, where="mid", label="cumulative explained variance")
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.legend(loc='best')
plt.show()

# 4. 特征对: 匹配特征值与特征向量, 并按特征值从大到小排序
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i]) for i in range(len(eigen_vals))]
eigen_pairs.sort(reverse=True)

# 5. 利用 top k 特征向量构造映射矩阵 W (此处, k=2)
w = np.hstack((eigen_pairs[0][1][:, np.newaxis],
               eigen_pairs[1][1][:, np.newaxis]))
print("Matrix W:\n", w)

# 6. 特征映射
X_train_pca = X_train_std.dot(w)

colors = ["r", "b", "g"]
markers = ["s", "x", "o"]
for l, c, m in zip(np.unique(y_train), colors, markers):
    plt.scatter(X_train_pca[y_train == l, 0],
                X_train_pca[y_train == l, 1],
                c=c, label=l, marker=m)

plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.show()


