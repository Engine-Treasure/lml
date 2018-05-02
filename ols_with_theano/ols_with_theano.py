import theano
import matplotlib.pyplot as plt
from theano import tensor as T
import numpy as np


def train_linre(X_train, y_train, eta, epochs):
    costs = []

    # init arrays
    eta0 = T.scalar("eta0")
    y = T.fvector(name="y")
    X = T.fmatrix(name="X")
    w = theano.shared(np.zeros(shape=(X_train.shape[1] + 1),
                               dtype=theano.config.floatX),
                      name="w")

    # calculate cost
    net_input = T.dot(X, w[1:]) + w[0]
    errors = y - net_input
    cost = T.sum(T.pow(errors, 2))

    # perform gradient update
    # 对 cost, 求 w 的微分. w 是一个矩阵, 将分别对每一个 w_j 求导
    gradient = T.grad(cost, wrt=w)
    update = [(w, w - eta0 * gradient)]

    # compile model
    train = theano.function(inputs=[eta0],
                            outputs=cost,
                            updates=update,
                            givens={X: X_train,
                                    y: y_train})

    for _ in range(epochs):
        # execution
        costs.append(train(eta))

    return costs, w


def predict_linreg(X, w):
    # defint
    Xt = T.matrix(name="X")
    net_input = T.dot(Xt, w[1:] + w[0])

    # compile
    predict = theano.function(inputs=[Xt],
                              givens={w: w},
                              outputs=net_input)

    # execute
    return predict(X)


X_train = np.asarray([[0.0], [1.0],
                      [2.0], [3.0],
                      [4.0], [5.0],
                      [6.0], [7.0],
                      [8.0], [9.0]],
                     dtype=theano.config.floatX)

y_train = np.asarray([1.0, 1.3,
                      3.1, 2.0,
                      5.0, 6.3,
                      6.6, 7.4,
                      8.0, 9.0],
                     dtype=theano.config.floatX)


costs, w = train_linre(X_train, y_train, eta=0.001, epochs=10)
plt.plot(range(1, len(costs) + 1), costs)
plt.tight_layout()
plt.xlabel("Epochs")
plt.ylabel("Cost")
plt.show()

plt.scatter(X_train,
            y_train,
            marker="s",
            s=50)
plt.plot(range(X_train.shape[0]),
         predict_linreg(X_train, w),
         color="red",
         marker="o",
         markersize=4,
         linewidth=3)
plt.xlabel("x")
plt.ylabel("y")
plt.show()
