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
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# 2. 为每个类计算 d 维 均值向量 (d 为特征数)
np.set_printoptions(precision=4)
mean_vecs = []
for label in range(1, 4):
    mean_vecs.append(np.mean(X_train_std[y_train == label], axis=0))
    print("MV %s: %s\n" % (label, mean_vecs[label - 1]))

print("Class Label Distribution: %s" % np.bincount(y)[1:])

# 3. 构造类内散布矩阵
d = X_train_std.shape[1]
S_W = np.zeros((d, d))
for label, mv in zip(range(1, 4), mean_vecs):
    # 对每个类, 计算类散布矩阵
    class_scatter = np.cov(X_train_std[y_train == label].T)
    # 求和
    S_W += class_scatter

print("Within-class scatter matrix: %sx%s" % (S_W.shape[0], S_W.shape[1]))

mean_overall = np.mean(X_train_std, axis=0)
S_B = np.zeros((d, d))
for i, mean_vec in enumerate(mean_vecs):
    n = X[y == i + 1, :].shape[0]
    mean_vec = mean_vec.reshape(d, 1)
    mean_overall = mean_overall.reshape(d, 1)
    S_B += n * (mean_vec - mean_overall).dot((mean_vec - mean_overall).T)

print("Between-class scatter matrix: %sx%s" % (S_B.shape[0], S_B.shape[1]))

# 4. 对 S_W.I * S_B 矩阵进行特征分解
eigen_vals, eigen_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))

# 选择 top k 个特征向量
eigen_pairs = [
    (np.abs(eigen_vals[i]), eigen_vecs[:, i])
    for i in range(len(eigen_vals))
]
eigen_pairs = sorted(eigen_pairs, key=lambda k: k[0], reverse=True)
print("Eigenvalues in decreasing order:\n")
for eigen_val in eigen_pairs:
    print(eigen_val[0])

# 绘图, 观察线性可判别性
tot = sum(eigen_vals.real)
discr = [(i / tot) for i in sorted(eigen_vals.real, reverse=True)]
cum_discr = np.cumsum(discr)
plt.bar(range(1, 14), discr, alpha=0.5, align="center", label="individual 'discriminability'")
plt.step(range(1, 14), cum_discr, where="mid", label="cumulative 'discriminability'")
plt.ylabel("'discriminability' ratio")
plt.xlabel("Linear Discriminability")
plt.ylim([-0.1, 1.1])
plt.legend(loc="best")
plt.show()

# 5. 构造转换矩阵
w = np.hstack((eigen_pairs[0][1][:, np.newaxis].real,
               eigen_pairs[1][1][:, np.newaxis].real))
print("Transform Matrix W:\n", w)

# 6. 转换
X_train_lda = X_train_std.dot(w)

colors = ["r", "b", "g"]
markers = ["s", "x", "o"]
for l, c, m in zip(np.unique(y_train), colors, markers):
    plt.scatter(X_train_lda[y_train == l, 0],
                X_train_lda[y_train == l, 1],
                c=c, label=l, marker=m)

plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='upper right')
plt.show()
