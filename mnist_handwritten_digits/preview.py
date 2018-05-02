import os
import struct

import numpy as np
import matplotlib.pyplot as plt


def load_mnist(path, kind="train"):
    labels_path = os.path.join(path, "{prefix}-labels.idx1-ubyte".format(prefix=kind))
    images_path = os.path.join(path, "{prefix}-images.idx3-ubyte".format(prefix=kind))

    with open(labels_path, "rb") as lbpath:
        # consume useless data
        magic, n = struct.unpack(">II", lbpath.read(8))
        # print(magic, n)

        labels = np.fromfile(lbpath, dtype=np.uint8)

    with open(images_path, "rb") as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII", imgpath.read(16))
        # print(magic, num, rows, cols)

        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)

    return images, labels


X_train, y_train = load_mnist(os.path.abspath("."))
X_test, y_test = load_mnist(os.path.abspath("."), kind="t10k")


# np.savetxt("train_img.csv", X_train, fmt="%i", delimiter=",")
# np.savetxt("train_labels.csv", y_train, fmt="%i", delimiter=",")
# np.savetxt("test_img.csv", X_train, fmt="%i", delimiter=",")
# np.savetxt("test_labels.csv", y_train, fmt="%i", delimiter=",")


# ax is an instance of np.ndarray
fig1, ax1 = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True)


ax1 = ax1.flatten()
for i in range(10):
    # X_train[y_train] - bool filter
    img = X_train[y_train == i][0].reshape(28, 28)
    ax1[i].imshow(img, cmap="Greys", interpolation="nearest")

ax1[0].set_xticks([])
ax1[0].set_yticks([])

fig1.tight_layout()

fig2, ax2 = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True)

ax2 = ax2.flatten()
for i in range(25):
    img = X_train[y_train == 7][i].reshape(28, 28)
    ax2[i].imshow(img, cmap="Greys", interpolation="nearest")

ax2[0].set_xticks([])
ax2[0].set_yticks([])

fig2.tight_layout()
plt.show()
