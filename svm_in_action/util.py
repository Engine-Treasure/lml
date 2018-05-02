# -*- coding: utf-8 -*-

from sklearn.preprocessing import LabelEncoder
import numpy as np

import matplotlib.pyplot as plt


def plot_svm_illustration(X, y, sv, w, b):
    X = np.mat(X)

    le = LabelEncoder()
    le.fit(np.unique(y))
    y = le.transform(y)
    plt.scatter([X[y == 1][:, 0]], [X[y == 1][:, 1]], color="red", marker="o", label="class 1")
    plt.scatter([X[y == 0][:, 0]], [X[y == 0][:, 1]], color="blue", marker="x", label="class 2")

    plt.scatter([sv[:, 0]], [sv[:, 1]], s=150, color="", edgecolors="black", marker="o", label="support vector")

    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend(loc="upper left")

    plt.show()
