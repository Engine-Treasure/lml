# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from perceptron_adaline.util import plot_decision_regions_2

iris = datasets.load_iris()

X = iris.data[:, [2, 3]]
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

sc = StandardScaler()
sc.fit(X_train)  # 估计样本均值与标准差
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

svm = SVC(kernel="linear", C=1.0, random_state=0)
svm.fit(X_train_std, y_train)

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

plot_decision_regions_2(X_combined_std,y_combined, classifier=svm, test_idx=range(105,150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.show()

svm2 = SVC(kernel="rbf", random_state=0, gamma=0.2, C=1.0)
svm2.fit(X_train_std, y_train)
plot_decision_regions_2(X_combined_std, y_combined, classifier=svm2, test_idx=range(105, 150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.show()

svm3 = SVC(kernel="rbf", random_state=0, gamma=100, C=1.0)
svm3.fit(X_train_std, y_train)
plot_decision_regions_2(X_combined_std, y_combined, classifier=svm3, test_idx=range(105, 150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.show()
