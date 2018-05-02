# -*- coding: utf-8 -*-

from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


class SBS():
    def __init__(self, estimator, k_features,
                 scoring=accuracy_score,
                 test_size=0.25, random_state=1):
        """

        :param estimator:
        :param k_features:  预期的特征数
        :param scoring:  用于性能评估
        :param test_size:
        :param random_state:
        :return:
        """
        self.scoring = scoring
        self.estimator = clone(estimator)
        self.k_features = k_features
        self.test_size = test_size
        self.random_state = random_state

    def fit(self, X, y):
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=self.test_size,
                             random_state=self.random_state)

        dim = X_train.shape[1]  # 维度
        self.indices_ = tuple(range(dim))  # 维度索引
        self.subsets_ = [self.indices_]  # 默认初始特征子集就是原始特征集本身
        # 使用原始特征集的性能, 精确度得分
        score = self._calc_score(X_train, y_train, X_test, y_test, self.indices_)

        self.scores_ = [score]

        while dim > self.k_features:
            scores = []
            subsets = []

            # 创建特征子集, combinations(self.indices_, r=dim - 1) 将返回 dim-1 的特征组合集
            for p in combinations(self.indices_, r=dim - 1):
                score = self._calc_score(X_train, y_train, X_test, y_test, p)

                scores.append(score)
                subsets.append(p)

            best = np.argmax(scores)  # np.argmax 是好函数, 返回最大值对应的索引
            self.indices_ = subsets[best]
            self.subsets_.append(self.indices_)
            dim -= 1

            self.scores_.append(scores[best])  # 保留最佳得分, 用于评估
        self.k_score_ = self.scores_[-1]

        return self

    def transform(self, X):
        return X[:, self.indices_]

    def _calc_score(self, X_train, y_train, X_test, y_test, indices):
        self.estimator.fit(X_train[:, indices], y_train)

        y_pred = self.estimator.predict(X_test[:, indices])
        score = self.scoring(y_test, y_pred)

        return score


if __name__ == '__main__':
    df_wine = pd.read_csv("wine.data", header=None)
    df_wine.columns = ['Class label', 'Alcohol',
                       'Malic acid', 'Ash',
                       'Alcalinity of ash', 'Magnesium',
                       'Total phenols', 'Flavanoids',
                       'Nonflavanoid phenols',
                       'Proanthocyanins',
                       'Color intensity', 'Hue',
                       'OD280/OD315 of diluted wines',
                       'Proline']

    X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values

    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.3, random_state=0)

    sc = StandardScaler()
    sc.fit(X_train)  # 估计样本均值与标准差
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    knn = KNeighborsClassifier(n_neighbors=2)
    sbs = SBS(knn, k_features=1)
    sbs.fit(X_train_std, y_train)

    k_feat = [len(k) for k in sbs.subsets_]
    plt.plot(k_feat, sbs.scores_, marker="o")
    plt.ylim([0.7, 1.1])
    plt.ylabel("Accuracy")
    plt.xlabel("Number of features")
    plt.grid()
    plt.show()

    # 第八次迭代 (不算第一次全特征), 使得特征只剩下 5 个
    k5 = list(sbs.subsets_[8])
    print(df_wine.columns[1:][k5])

    knn.fit(X_train_std, y_train)
    print("Training accuracy:", knn.score(X_train_std, y_train))
    print("Test accuracy:", knn.score(X_test_std, y_test))

    knn.fit(X_train_std[:, k5], y_train)
    print("Training accuracy of 5 features:", knn.score(X_train_std[:, k5], y_train))
    print("Test accuracy of 5 features:", knn.score(X_test_std[:, k5], y_test))

