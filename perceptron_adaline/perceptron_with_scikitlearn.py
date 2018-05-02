# -*- coding: utf-8 -*-

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

from perceptron_adaline.util import plot_decision_regions_2

iris = datasets.load_iris()

X = iris.data[:, [2, 3]]
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

sc = StandardScaler()
sc.fit(X_train)  # 估计样本均值与标准差
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# random_state 是每轮洗牌的标志
ppn = Perceptron(n_iter=40, eta0=0.1, random_state=0)
ppn.fit(X_train_std, y_train)

y_pred = ppn.predict(X_test_std)
print("Misclassified samples: %d" % (y_pred != y_test).sum())
print("Accuracy: %.2f" % accuracy_score(y_test, y_pred))

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

plot_decision_regions_2(X=X_combined_std,
                        y=y_combined,
                        classifier=ppn,
                        test_idx=range(105, 150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.show()
