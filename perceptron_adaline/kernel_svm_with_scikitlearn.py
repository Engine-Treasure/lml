# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from perceptron_adaline.util import plot_decision_regions

np.random.seed(0)

X_xor = np.random.randn(200, 2)
print(X_xor)
y_xor = np.logical_xor(X_xor[:, 0] > 0, X_xor[:, 1] > 0)
print(y_xor)
y_xor = np.where(y_xor, 1, -1)
print(y_xor)

svm = SVC(kernel="rbf", random_state=0, gamma=0.1, C=10.0)
svm.fit(X_xor, y_xor)

plot_decision_regions(X_xor, y_xor, classifier=svm)
plt.legend(loc="upper left")
plt.show()

