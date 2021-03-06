# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class LinearRegressionGD(object):
    def __init__(self, eta=0.001, n_iter=20):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        self.w_ = np.zeros((1 + X.shape[1], 1))
        self.cost_ = []

        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = (y - output)
            self.w_[0] += self.eta * errors.sum()
            self.w_[1:] += self.eta * np.dot(X.T, errors)
            cost = (errors ** 2).sum() / 2.0
            self.cost_.append(cost)

        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return self.net_input(X)


def lin_regplot(X, y, model):
    plt.scatter(X, y, c="blue")
    plt.plot(X, model.predict(X), color="red")
    return None


if __name__ == '__main__':
    df = pd.read_csv("housing.data", header=None, sep="\s+")
    df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS',
                  'NOX', 'RM', 'AGE', 'DIS', 'RAD',
                  'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    X = df[["RM"]].values
    y = df[["MEDV"]].values
    from sklearn.preprocessing import StandardScaler

    sc_x = StandardScaler()
    sc_y = StandardScaler()
    X_std = sc_x.fit_transform(X)
    y_std = sc_y.fit_transform(y)

    lr = LinearRegressionGD()
    lr.fit(X_std, y_std)

    plt.plot(range(1, lr.n_iter + 1), lr.cost_)
    plt.ylabel("SSE")
    plt.xlabel("Epoch")
    plt.show()

    lin_regplot(X_std, y_std, lr)
    plt.xlabel("average number of rooms [rm] (standardized)")
    plt.ylabel("price in $1000\'s [medv] (standardized)")
    plt.show()

    num_rooms_std = sc_x.transform([5.0])
    price_std = lr.predict(num_rooms_std)
    print("Price in $1000's: %.3f" % sc_y.inverse_transform(price_std))

    print("Slope: %.3f" % lr.w_[1])
    print("Intercept: %.3f" % lr.w_[0])
