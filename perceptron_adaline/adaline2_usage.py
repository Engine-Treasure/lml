import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from perceptron_adaline.adaline2 import AdalineSGD
from perceptron_adaline.util import plot_decision_regions

df = pd.read_csv("perceptron_adaline.data", header=None)

y = df.iloc[0: 100, 4].values
y = np.where(y == "Iris-setosa", -1, 1)
X = df.iloc[0: 100, [0, 2]].values

X_std = np.copy(X)
X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()

ada = AdalineSGD(n_iter=15, eta=0.01, random_state=1)
ada.fit(X, y)

plot_decision_regions(X_std, y, classifier=ada)
plt.xlabel("sepal length [standardlized]")
plt.ylabel("petal length [standardlized]")
plt.legend(loc="upper left")
plt.show()

plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker="o")
plt.xlabel("Epochs")
plt.ylabel("Average Cost")
plt.show()

