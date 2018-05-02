# -*- coding: utf-8 -*-

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

from decision_tree_in_action_id3.util import plot_decision_regions_2

iris = datasets.load_iris()

X = iris.data[:, [2, 3]]
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

forest = RandomForestClassifier(criterion="entropy",
                                n_estimators=10,  # 决策树数
                                random_state=1,
                                n_jobs=2)  # 并行
forest.fit(X_train, y_train)

X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))

plot_decision_regions_2(X_combined, y_combined, classifier=forest, test_idx=range(105, 150))

plt.xlabel("petal length [cm]")
plt.ylabel("petal width [cm]")
plt.legend(loc="upper left")
plt.show()
