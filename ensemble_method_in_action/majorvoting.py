# -*- coding: utf-8 -*-

import operator

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.preprocessing import LabelEncoder
from sklearn.externals import six  # 兼容性顾忌
from sklearn.pipeline import _name_estimators

# 加权大多数投票, 第一个参数是投票结果, 第二个参数是权值
np.argmax(np.bincount([0, 0, 1], weights=[0.2, 0.2, 0.6]))

# 基于概率的加权投票结果, 示例
ex = np.array([[0.9, 0.1], [0.8, 0.2], [0.4, 0.6]])  # 每个分类器预测的概率分布
p = np.average(ex, axis=0, weights=[0.2, 0.2, 0.6])  #
np.argmax(p)


class MajorityVoteClassifier(BaseEstimator, ClassifierMixin):
    """ A majority vote ensemble classifier

    Parameters
    ----------
    classifiers : array-like, shape = [n_classifiers]
        Different classifiers for the ensemble

    vote : str, {'classlabel', 'probability'}
        Default: 'classlabel'
        If 'classlabel' the prediction is based on
        the argmax of class labels. Else if
        'probability', the argmax of the sum of
        probabilities is used to predict the class label
        (recommended for calibrated classifiers).

    weights : array-like, shape = [n_classifiers]
        Optional, default: None
        If a list of `int` or `float` values are
        provided, the classifiers are weighted by
        importance; Uses uniform weights if `weights=None`.

    """

    def __init__(self, classifiers, vote="classlabel", weights=None):
        self.classifiers = classifiers
        # _name_estimators - 为 estimators 生成名称
        self.named_classifiers = {key: value for key, value in _name_estimators(classifiers)}

        self.vote = vote
        self.weights = weights

    def fit(self, X, y):
        """ Fit classifiers.

        Parameters
        ----------
        X : {array-like, sparse matrix},
            shape = [n_samples, n_features]
            Matrix of training samples.

        y : array-like, shape = [n_samples]
            Vector of target class labels.

        Returns
        -------
        self : object

        """

        # 使用 LabelEncoder, 是为了确保类标签从 0 开始, 这对于 np.argmax 很关键
        self.labelenc_ = LabelEncoder()
        self.labelenc_.fit(y)
        self.classes_ = self.labelenc_.classes_
        self.classifiers_ = []
        for clf in self.classifiers:
            fitted_clf = clone(clf).fit(X, self.labelenc_.transform(y))
            self.classifiers_.append(fitted_clf)

        return self

    def predict(self, X):
        """ Predict class labels for X.

        Parameters
        ----------
        X : {array-like, sparse matrix},
            Shape = [n_samples, n_features]
            Matrix of training samples.

        Returns
        ----------
        maj_vote : array-like, shape = [n_samples]
            Predicted class labels.

        """
        if self.vote == "probability":
            maj_vote = np.argmax(self.predict_proba(X), axis=1)
        else:  # 对类标签投票
            # 收集更分类器的预测结果
            predictions = np.asarray([
                clf.predict(X) for clf in self.classifiers_
            ]).T
            maj_vote = np.apply_along_axis(
                lambda x: np.argmax(np.bincount(x, weights=self.weights)),
                axis=1,
                arr=predictions
            )
            maj_vote = self.labelenc_.inverse_transform(maj_vote)

        return maj_vote

    def predict_proba(self, X):
        """ Predict class probabilities for X.

        Parameters
        ----------
        X : {array-like, sparse matrix},
            shape = [n_samples, n_features]
            Training vectors, where n_samples is
            the number of samples and
            n_features is the number of features.

        Returns
        ----------
        avg_proba : array-like,
            shape = [n_samples, n_classes]
            Weighted average probability for
            each class per sample.

        """
        probas = np.asarray([
            clf.predict_proba(X) for clf in self.classifiers_
        ])
        avg_proba = np.average(probas, axis=0, weights=self.weights)

        return avg_proba

    def get_params(self, deep=True):
        """ Get classifier parameter names for GridSearch"""
        if not deep:
            return super(MajorityVoteClassifier, self).get_params(deep=False)
        else:
            out = self.named_classifiers.copy()
            for name, step in six.iteritems(self.named_classifiers):
                for k, v in six.iteritems(step.get_params(deep=True)):
                    out["%s__%s" % (name, k)] = v

            return out
