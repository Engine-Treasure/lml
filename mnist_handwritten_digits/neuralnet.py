# -*- coding: utf-8 -*-
import os
import struct
import sys

import numpy as np
from scipy.special import expit


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


class NeuralNetMLP(object):
    def __init__(self, n_output, n_features, n_hidden=30, l1=0.0, l2=0.0, epochs=500, eta=0.001,
                 alpha=0.0, decrease_const=0.0, shuffle=True, minibatches=1, random_state=None):
        np.random.seed(random_state)
        self.n_output = n_output  # 输出层神经元数
        self.n_features = n_features  # 特征数, 输入神经元数
        self.n_hidden = n_hidden  # 隐层神经元数
        self.w1, self.w2 = self._initialize_weights()  # 连接权

        # 减小过拟合的程度
        self.l1 = l1  # L1 范数正则化
        self.l2 = l2  # L2 范数正则化

        self.epochs = epochs  # 迭代次数
        self.eta = eta  # 学习率
        self.alpha = alpha  # ∆w_t = η∇J(w_t) + α∆w_t−1, 前一次梯度的权值, 为更快的学习
        self.decrease_const = decrease_const  # 衰减常数, η/(1 + t×d) (t 为迭代的代次)
        self.shuffle = shuffle  # 单词本意是: 洗牌. 每次迭代之前先清洗训练集, 避免陷入循环
        self.minibatches = minibatches  # 最小批量. 每次迭代, 将训练集分成 k 个 最小批量, 为每个最小批量独立计算梯度, 以更快的学习

    def _encode_labels(self, y, k):
        """Encode labels into one-hot representation

        Parameters
        ------------
        y: array, shape = [n_samples]
            Target values.

        Returns
        -----------
        onehot : array, shape = (n_labels, n_samples)

        """
        onehot = np.zeros((k, y.shape[0]))
        for idx, val in enumerate(y):
            onehot[val, idx] = 1.0
        return onehot

    def _initialize_weights(self):
        # bias unit
        # 随机初始化
        # Initializing all theta weights to zero does not work with neural networks
        w1 = np.random.uniform(-1.0, 1.0, size=self.n_hidden * (self.n_features + 1))
        w1 = w1.reshape(self.n_hidden, self.n_features + 1)
        w2 = np.random.uniform(-1.0, 1.0, size=self.n_output * (self.n_hidden + 1))
        w2 = w2.reshape(self.n_output, self.n_hidden + 1)

        return w1, w2

    def _sigmoid(self, z):
        # known as the logistic function, is defined as expit(x) = 1 / (1 + exp(-x))
        return expit(z)

    def _sigmoid_gradient(self, z):
        """
        Compute gradient of the logistic function
        g prime, or g'
        """
        sg = self._sigmoid(z)
        return sg * (1 - sg)

    def _add_bias_unit(self, X, how="column"):
        if how == "column":
            X_new = np.ones((X.shape[0], X.shape[1] + 1))
            X_new[:, 1:] = X
        elif how == "row":
            X_new = np.ones((X.shape[0] + 1, X.shape[1]))
            X_new[1:, :] = X
        else:
            raise AttributeError("`how` must be `column` or `row`")
        return X_new

    def _feedforward(self, X, w1, w2):
        a1 = self._add_bias_unit(X, how='column')
        z2 = w1.dot(a1.T)
        a2 = self._sigmoid(z2)
        a2 = self._add_bias_unit(a2, how='row')
        z3 = w2.dot(a2)
        a3 = self._sigmoid(z3)
        return a1, z2, a2, z3, a3

    def _L2_reg(self, lambda_, w1, w2):
        # Do not regularize the bias unit
        return (lambda_ / 2.0) * (np.sum(w1[:, 1:] ** 2) + np.sum(w2[:, 1:] ** 2))

    def _L1_reg(self, lambda_, w1, w2):
        # Do not regularize the bias unit
        return (lambda_ / 2.0) * (np.abs(w1[:, 1:]).sum() + np.abs(w2[:, 1:]).sum())

    def _get_cost(self, y_enc, output, w1, w2):
        """Compute cost function.

        Parameters
        ----------
        y_enc : array, shape = (n_labels, n_samples)
            one-hot encoded class labels.
        output : array, shape = [n_output_units, n_samples]
            Activation of the output layer (feedforward)
        w1 : array, shape = [n_hidden_units, n_features]
            Weight matrix for input layer -> hidden layer.
        w2 : array, shape = [n_output_units, n_hidden_units]
            Weight matrix for hidden layer -> output layer.

        Returns
        ---------
        cost : float
            Regularized cost.

        """
        # logistic cost function
        term1 = -y_enc * (np.log(output))
        term2 = (1 - y_enc) * np.log(1 - output)
        cost = np.sum(term1 - term2)

        # regularize term, to reduce the degree of overfitting
        L1_term = self._L1_reg(self.l1, w1, w2)
        L2_term = self._L2_reg(self.l2, w1, w2)

        # cost with 2 regularization items
        cost = cost + L1_term + L2_term
        return cost

    def _get_gradient(self, a1, a2, a3, z2, y_enc, w1, w2):
        # backpropagation
        sigma3 = a3 - y_enc
        z2 = self._add_bias_unit(z2, how='row')
        sigma2 = w2.T.dot(sigma3) * self._sigmoid_gradient(z2)
        sigma2 = sigma2[1:, :]
        grad1 = sigma2.dot(a1)
        grad2 = sigma3.dot(a2.T)

        # regularize
        grad1[:, 1:] += (w1[:, 1:] * (self.l1 + self.l2))
        grad2[:, 1:] += (w2[:, 1:] * (self.l1 + self.l2))
        return grad1, grad2

    def predict(self, X):
        a1, z2, a2, z3, a3 = self._feedforward(X, self.w1, self.w2)

        y_pred = np.argmax(z3, axis=0)
        return y_pred

    def fit(self, X, y, print_progress=False):
        self.cost_ = []

        X_data, y_data = X.copy(), y.copy()
        y_enc = self._encode_labels(y, self.n_output)

        # 前一次的权值梯度
        delta_w1_prev = np.zeros(self.w1.shape)
        delta_w2_prev = np.zeros(self.w2.shape)

        for i in range(self.epochs):
            # adaptive learning rate
            self.eta /= (1 + self.decrease_const * i)

            if print_progress:
                sys.stderr.write('\rEpoch: %d/%d' % (i + 1, self.epochs))
                sys.stderr.flush()

            if self.shuffle:
                # permutation: 排列
                # 随机排列, 返回打乱的索引
                idx = np.random.permutation(y_data.shape[0])
                X_data, y_data = X_data[idx], y_data[idx]

            # 将数组划分成 50 个一组的小数组
            mini = np.array_split(range(y_data.shape[0]), self.minibatches)
            for idx in mini:
                # feedforward
                a1, z2, a2, z3, a3 = self._feedforward(X[idx], self.w1, self.w2)
                cost = self._get_cost(y_enc=y_enc[:, idx],
                                      output=a3,
                                      w1=self.w1,
                                      w2=self.w2)
                self.cost_.append(cost)
                # compute gradient via backpropagation
                grad1, grad2 = self._get_gradient(a1=a1, a2=a2,
                                                  a3=a3, z2=z2,
                                                  y_enc=y_enc[:, idx],
                                                  w1=self.w1,
                                                  w2=self.w2)

                # update weights
                delta_w1, delta_w2 = self.eta * grad1, self.eta * grad2
                self.w1 -= (delta_w1 + (self.alpha * delta_w1_prev))
                self.w2 -= (delta_w2 + (self.alpha * delta_w2_prev))
                delta_w1_prev, delta_w2_prev = delta_w1, delta_w2

        return self


class MLPGradientCheck(NeuralNetMLP):
    def __init__(self, n_output, n_features, n_hidden=30, l1=0.0, l2=0.0, epochs=500, eta=0.001,
                 alpha=0.0, decrease_const=0.0, shuffle=True, minibatches=1, random_state=None):

        super(MLPGradientCheck, self).__init__(n_output, n_features, n_hidden, l1, l2, epochs, eta,
                                               alpha, decrease_const, shuffle, minibatches, random_state)

    def _gradient_checking(self, X, y_enc, w1, w2, epsilon, grad1, grad2):
        """gradient checking (for debugging only)

        Returns
        ---------
        relative_error : float
        Relative error between the numerically approximated gradients and the backpropagated gradients.
        """
        num_grad1 = np.zeros(np.shape(w1))
        epsilon_ary1 = np.zeros(np.shape(w1))
        for i in range(w1.shape[0]):
            for j in range(w1.shape[1]):
                epsilon_ary1[i, j] = epsilon
                a1, z2, a2, z3, a3 = self._feedforward(X, w1 - epsilon_ary1, w2)
                cost1 = self._get_cost(y_enc, a3, w1 - epsilon_ary1, w2)
                a1, z2, a2, z3, a3 = self._feedforward(X, w1 + epsilon_ary1, w2)
                cost2 = self._get_cost(y_enc, a3, w1 + epsilon_ary1, w2)
                num_grad1[i, j] = (cost2 - cost1) / (2 * epsilon)
                epsilon_ary1[i, j] = 0

        num_grad2 = np.zeros(np.shape(w2))
        epsilon_ary2 = np.zeros(np.shape(w2))
        for i in range(w2.shape[0]):
            for j in range(w2.shape[1]):
                epsilon_ary2[i, j] = epsilon
                a1, z2, a2, z3, a3 = self._feedforward(X, w1, w2 - epsilon_ary2)
                cost1 = self._get_cost(y_enc, a3, w1, w2 - epsilon_ary2)
                a1, z2, a2, z3, a3 = self._feedforward(X, w1, w2 + epsilon_ary2)
                cost2 = self._get_cost(y_enc, a3, w1, w2 + epsilon_ary2)
                num_grad2[i, j] = (cost2 - cost1) / (2 * epsilon)
                epsilon_ary2[i, j] = 0

        num_grad = np.hstack((num_grad1.flatten(),
                              num_grad2.flatten()))
        grad = np.hstack((grad1.flatten(), grad2.flatten()))
        norm1 = np.linalg.norm(num_grad - grad)
        norm2 = np.linalg.norm(num_grad)
        norm3 = np.linalg.norm(grad)
        relative_error = norm1 / (norm2 + norm3)

        return relative_error

    def fit(self, X, y, print_progress=False):
        self.cost_ = []

        X_data, y_data = X.copy(), y.copy()
        y_enc = self._encode_labels(y, self.n_output)

        # 前一次的权值梯度
        delta_w1_prev = np.zeros(self.w1.shape)
        delta_w2_prev = np.zeros(self.w2.shape)

        for i in range(self.epochs):
            # adaptive learning rate
            self.eta /= (1 + self.decrease_const * i)

            if print_progress:
                sys.stderr.write('\rEpoch: %d/%d' % (i + 1, self.epochs))
                sys.stderr.flush()

            if self.shuffle:
                # permutation: 排列
                # 随机排列, 返回打乱的索引
                idx = np.random.permutation(y_data.shape[0])
                X_data, y_data = X_data[idx], y_data[idx]

            # 将数组划分成 50 个一组的小数组
            mini = np.array_split(range(y_data.shape[0]), self.minibatches)
            for idx in mini:
                # feedforward
                a1, z2, a2, z3, a3 = self._feedforward(X[idx], self.w1, self.w2)
                cost = self._get_cost(y_enc=y_enc[:, idx],
                                      output=a3,
                                      w1=self.w1,
                                      w2=self.w2)
                self.cost_.append(cost)
                # compute gradient via backpropagation
                grad1, grad2 = self._get_gradient(a1=a1, a2=a2,
                                                  a3=a3, z2=z2,
                                                  y_enc=y_enc[:, idx],
                                                  w1=self.w1,
                                                  w2=self.w2)


                # start gradient checking
                grad_diff = self._gradient_checking(X=X[idx], y_enc=y_enc[:, idx],
                                                    w1=self.w1, w2=self.w2, epsilon=1e-5,
                                                    grad1=grad1, grad2=grad2)
                if grad_diff <= 1e-7:
                    print("Ok: {grad_diff}".format(grad_diff=grad_diff))
                elif grad_diff <= 1e-4:
                    print("Warning: {grad_diff}".format(grad_diff=grad_diff))
                else:
                    print("PROBLEM: {grad_diff}".format(grad_diff=grad_diff))
                # end gradient checking


                # update weights
                delta_w1, delta_w2 = self.eta * grad1, self.eta * grad2
                self.w1 -= (delta_w1 + (self.alpha * delta_w1_prev))
                self.w2 -= (delta_w2 + (self.alpha * delta_w2_prev))
                delta_w1_prev, delta_w2_prev = delta_w1, delta_w2

        return self
