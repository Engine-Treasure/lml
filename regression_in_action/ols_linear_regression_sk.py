# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression, RANSACRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures


def lin_regplot(X, y, model):
    plt.scatter(X, y, c="blue")
    plt.plot(X, model.predict(X), color="red")
    return None


df = pd.read_csv("housing.data", header=None, sep="\s+")
df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS',
              'NOX', 'RM', 'AGE', 'DIS', 'RAD',
              'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
X = df[["RM"]].values
y = df[["MEDV"]].values

# Linear Regression
slr = LinearRegression()
slr.fit(X, y)
print("Slope: %.3f" % slr.coef_[0])
print("Intercept: %.3f" % slr.intercept_)

lin_regplot(X, y, slr)
plt.xlabel("average number of rooms [rm] (standardized)")
plt.ylabel("price in $1000\'s [medv] (standardized)")
plt.show()

# RANSAC
ransac = RANSACRegressor(LinearRegression(),
                         max_trials=100,
                         min_samples=50,
                         residual_metric=lambda x: np.sum(np.abs(x), axis=1),
                         residual_threshold=5.0,
                         random_state=0
                         )

ransac.fit(X, y)
print("Slope: %.3f" % ransac.estimator_.coef_[0])
print("Intercept: %.3f" % ransac.estimator_.intercept_)

inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)
line_X = np.arange(3, 10, 1)
line_Y_ransac = ransac.predict(line_X[:, np.newaxis])
plt.scatter(X[inlier_mask], y[inlier_mask], c="blue", marker="o", label="Inliers")
plt.scatter(X[outlier_mask], y[outlier_mask], c="green", marker="s", label="Outliers")
plt.plot(line_X, line_Y_ransac, color="red")
plt.xlabel("average number of rooms [rm]")
plt.ylabel("price in $1000\'s [medv]")
plt.legend(loc="upper left")
plt.show()

# MSE, R^2
X = df.iloc[:, :-1].values
y = df["MEDV"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

slr = LinearRegression()
slr.fit(X_train, y_train)
y_train_pred = slr.predict(X_train)
y_test_pred = slr.predict(X_test)
print("MSE train: %.3f, test: %.3f" % (
    mean_squared_error(y_train, y_train_pred),
    mean_squared_error(y_test, y_test_pred),
))
print("R^2 train: %.3f, test: %.3f" % (
    r2_score(y_train, y_train_pred),
    r2_score(y_test, y_test_pred),
))

plt.scatter(y_train_pred, y_train_pred - y_train, c="blue", marker="o", label="Training data")
plt.scatter(y_test_pred, y_test_pred - y_test, c="lightgreen", marker="s", label="Test data")
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.legend(loc="upper left")
plt.hlines(y=0, xmin=-10, xmax=50, lw=2, color="red")
plt.xlim([-10, 50])
plt.show()

# Polynomial Regression Demo
X = np.array([258.0, 270.0, 294.0,
              320.0, 342.0, 368.0,
              396.0, 446.0, 480.0,
              586.0])[:, np.newaxis]
y = np.array([236.4, 234.4, 252.8,
              298.6, 314.2, 342.2,
              360.8, 368.0, 391.2,
              390.8])
lr = LinearRegression()
pr = LinearRegression()
quadratic = PolynomialFeatures(degree=2)
X_quad = quadratic.fit_transform(X)

lr.fit(X, y)
X_fit = np.arange(250, 600, 10)[:, np.newaxis]  # 用于预测
y_lin_fit = lr.predict(X_fit)

pr.fit(X_quad, y)
y_quad_fit = pr.predict(quadratic.fit_transform(X_fit))

plt.scatter(X, y, label="training points")
plt.plot(X_fit, y_lin_fit, label="linear fit", linestyle="--")
plt.plot(X_fit, y_quad_fit, label="quadratic fit")
plt.legend(loc="upper left")
plt.show()

y_lin_pred = lr.predict(X)  # X 和 X_quad 不是用于训练的嘛
y_quad_pred = pr.predict(X_quad)
print("Training MSE linear: %.3f, quadratic: %.3f" % (mean_squared_error(y, y_lin_pred), mean_squared_error(y, y_quad_pred)))
print("Training R^2 linear: %.3f, quadratic: %.3f" % (r2_score(y, y_lin_pred), r2_score(y, y_quad_pred)))

# Modeling nonlinear relationships

X = df[["LSTAT"]].values
y = df[["MEDV"]].values
regr = LinearRegression()

# create polynomial features
quadratic = PolynomialFeatures(degree=2)
cubic = PolynomialFeatures(degree=3)
X_quad = quadratic.fit_transform(X)
X_cubic = cubic.fit_transform(X)

# linear fit
X_fit = np.arange(X.min(), X.max(), 1)[:, np.newaxis]
regr = regr.fit(X, y)
y_lin_fit = regr.predict(X_fit)
linear_r2 = r2_score(y, regr.predict(X))

# quadratic fit
regr = regr.fit(X_quad, y)
y_quad_fit = regr.predict(quadratic.fit_transform(X_fit))
quadratic_r2 = r2_score(y, regr.predict(X_quad))

# cubic fit
regr = regr.fit(X_cubic, y)
y_cubic_fit = regr.predict(cubic.fit_transform(X_fit))
cubic_r2 = r2_score(y, regr.predict(X_cubic))

# plot results
plt.scatter(X, y, label="training points", color="lightgray")
plt.plot(X_fit, y_lin_fit, label="linear (d=1), $R^2=%.2f" % linear_r2, color="blue", lw=2, linestyle=":")
plt.plot(X_fit, y_quad_fit, label="quadratic (d=2), $R^2=%.2f" % quadratic_r2, color="red", lw=2, linestyle="-")
plt.plot(X_fit, y_cubic_fit, label="cubic (d=3), $R^2=%.2f" % cubic_r2, color="green", lw=2, linestyle="--")
plt.xlabel("% lower status of the population [LSTAT]")
plt.ylabel("Price in %1000\'s [MEDV]")
plt.legend(loc="upper right")
plt.show()

# Yet another modeling nonlinear relationship

# transform features
X_log = np.log(X)
y_sqrt = np.sqrt(y)

# fit features
X_fit = np.arange(X_log.min() - 1, X_log.max() + 1, 1)[:, np.newaxis]
regr = regr.fit(X_log, y_sqrt)
y_lin_fit = regr.predict(X_fit)
linear_r2 = r2_score(y_sqrt, regr.predict(X_log))

# plot results
plt.scatter(X_log, y_sqrt, label="training points", color="lightgray")
plt.plot(X_fit, y_lin_fit, label="linear (d=1), $R^2=%.2f" % linear_r2, color="blue", lw=2)
plt.xlabel("% log(% lower status of the population [LSTAT])")
plt.ylabel("$\sqrt{Price in %1000\'s [MEDV]}")
plt.legend(loc="lower left")
plt.show()


