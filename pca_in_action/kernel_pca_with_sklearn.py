# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from sklearn.decomposition import KernelPCA
from sklearn.datasets import make_moons

X, y = make_moons(n_samples=100, random_state=123)
sklearn_kpca = KernelPCA(n_components=2, kernel="rbf", gamma=15)
X_skernpa = sklearn_kpca.fit_transform(X)

plt.scatter(X_skernpa[y == 0, 0], X_skernpa[y == 0, 1], color="red", marker="^", alpha=0.5)
plt.scatter(X_skernpa[y == 1, 0], X_skernpa[y == 1, 1], color="blue", marker="o", alpha=0.5)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()
