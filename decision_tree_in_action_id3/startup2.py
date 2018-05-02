# -*- coding: utf-8 -*-

from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

from decision_tree_in_action_id3.util import plot_decision_regions_2

iris = datasets.load_iris()

X = iris.data[:, [2, 3]]
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


tree = DecisionTreeClassifier(criterion="entropy",
                              max_depth=3,
                              random_state=0)
tree.fit(X_train, y_train)

X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))

plot_decision_regions_2(X_combined, y_combined, classifier=tree, test_idx=range(105, 150))

plt.xlabel("petal length [cm]")
plt.ylabel("petal width [cm]")
plt.legend(loc="upper left")
plt.show()

export_graphviz(tree, out_file="tree.dot", feature_names=["petal length", "petal width"])
