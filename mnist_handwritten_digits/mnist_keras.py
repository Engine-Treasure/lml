# -*- coding: utf-8 -*-
import os
import struct

import numpy as np
import theano
from keras.layers.core import Dense
from keras.models import Sequential
from keras.optimizers import SGD
from keras.utils import np_utils


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


theano.config.floatX = "float32"
X_train, y_train = load_mnist(os.path.abspath("."))
X_test, y_test = load_mnist(os.path.abspath("."), kind="t10k")

X_train = X_train.astype(theano.config.floatX)
X_test = X_test.astype(theano.config.floatX)

y_train_ohe = np_utils.to_categorical(y_train)

np.random.seed(1)

# implement a feedforward neural network
model = Sequential()

model.add(Dense(input_dim=X_train.shape[1],
                output_dim=50,
                init="uniform",
                activation="tanh"))

model.add(Dense(input_dim=50,
                output_dim=50,
                init="uniform",
                activation="tanh"))

model.add(Dense(input_dim=50,
                output_dim=y_train_ohe.shape[1],
                init="uniform",
                activation="softmax"))

# decay - 衰退, 每一轮迭代, 学习率的衰减量
# momentum - 动量
sgd = SGD(lr=0.001, decay=1e-7, momentum=0.9)
model.compile(loss="categorical_crossentropy", optimizer=sgd)

model.fit(X_train,
          y_train_ohe,
          nb_epoch=50,
          batch_size=300,
          verbose=1,
          validation_split=0.3)

y_train_pred = model.predict_classes(X_train, verbose=0)
train_acc = np.sum(y_train == y_train_pred, axis=0) / X_train.shape[0]
print("Training accuracy: {acc}".format(acc=train_acc * 100))

y_test_pred = model.predict_classes(X_test, verbose=0)
test_acc = np.sum(y_test == y_test_pred, axis=0) / X_test.shape[0]
print("Testing accuracy: {acc}".format(acc=test_acc * 100))
