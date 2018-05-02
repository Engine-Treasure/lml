# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

from pca_in_action.util import plot_decision_regions

df_wine = pd.read_csv("wine.data")
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

sc =StandardScaler()
lda = LinearDiscriminantAnalysis(n_components=2)
lr = LogisticRegression()

sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

lda.fit(X_train_std, y_train)
X_train_lda = lda.transform(X_train_std)
X_test_lda = lda.transform(X_test_std)

lr.fit(X_train_lda, y_train)
plot_decision_regions(X_train_lda, y_train, classifier=lr)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend(loc='lower left')
plt.show()


plot_decision_regions(X_test_lda, y_test, classifier=lr)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend(loc='lower left')
plt.show()

lda = LinearDiscriminantAnalysis(n_components=None)  # 设置为 None, 将保留所有的主成分
X_train_lda = lda.fit_transform(X_train_std, y_train)
print(lda.explained_variance_ratio_)  # 解释方差比率, 不知道什么东西
