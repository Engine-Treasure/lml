import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from perceptron_adaline.perceptron import Perceptron
from perceptron_adaline.util import plot_decision_regions

df = pd.read_csv("perceptron_adaline.data", header=None)

y = df.iloc[0: 100, 4].values
y = np.where(y == "Iris-setosa", -1, 1)
X = df.iloc[0: 100, [0, 2]].values
plt.scatter(X[: 50, 0], X[: 50, 1], color="red", marker="o", label="setosa")
plt.scatter(X[50: 100, 0], X[50: 100, 1], color="blue", marker="x", label="versicolor")

plt.xlabel("petal length")
plt.ylabel("sepal length")
plt.legend(loc="upper left")

plt.show()

ppn = Perceptron(eta=0.1, n_iter=10)

ppn.fit(X, y)

plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker="o")

plt.xlabel("Epochs")
plt.ylabel("Number of misclassification")

plt.show()


plot_decision_regions(X, y, classifier=ppn)
plt.xlabel("sepal length [cm]")
plt.ylabel("petal length [cm]")
plt.legend(loc="upper left")
plt.show()
