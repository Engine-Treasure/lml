import os

import numpy as np
import matplotlib.pyplot as plt

from mnist_handwritten_digits.neuralnet import load_mnist, NeuralNetMLP

X_train, y_train = load_mnist(os.path.abspath("."))
X_test, y_test = load_mnist(os.path.abspath("."), kind="t10k")

nn = NeuralNetMLP(n_output=10,
                  n_features=X_train.shape[1],
                  n_hidden=50,
                  l2=0.1,
                  l1=0.0,
                  epochs=1000,
                  eta=0.001,
                  alpha=0.001,
                  decrease_const=0.00001,
                  shuffle=True,
                  minibatches=50,
                  random_state=1)

nn.fit(X_train, y_train, print_progress=True)

plt.plot(range(len(nn.cost_)), nn.cost_)
plt.ylim([0, 2000])
plt.ylabel("Cost")
plt.xlabel("Epochs * 50")
plt.tight_layout()
plt.show()

batches = np.array_split(range(len(nn.cost_)), 1000)
cost_ary = np.array(nn.cost_)
cost_avgs = [np.mean(cost_ary[i]) for i in batches]

plt.plot(range(len(cost_avgs)), cost_avgs, color="red")
plt.ylim([0, 2000])
plt.ylabel("Cost")
plt.xlabel("Epochs")
plt.tight_layout()
plt.show()

y_train_pred = nn.predict(X_train)
acc = np.sum(y_train == y_train_pred, axis=0) / X_train.shape[0]
print('Training accuracy: %.2f%%' % (acc * 100))

y_test_pred = nn.predict(X_test)
acc = np.sum(y_test == y_test_pred, axis=0) / X_test.shape[0]
print('Training accuracy: %.2f%%' % (acc * 100))

miscl_img = X_test[y_test != y_test_pred][:25]
correct_lab = y_test[y_test != y_test_pred][:25]
miscl_lab = y_test_pred[y_test != y_test_pred][:25]

fig, ax = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True)

ax = ax.flatten()

for i in range(25):
    img = miscl_img[i].reshape(28, 28)
    ax[i].imshow(img, cmap="Greys",
                 interpolation="nearest")
    ax[i].set_title("{i} t: {t} p: {p}".format(i=i+1, t=correct_lab[i], p=miscl_lab[i]))

ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()
