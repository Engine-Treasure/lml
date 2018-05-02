# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

le = LabelEncoder()

df = pd.read_csv("wdbc.data", header=None)
X = df.loc[:, 2:].values
y = df.loc[:, 1].values
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Pipeline 对象接受元组的列表作为输入,
# 每个元组的第一个元素是 transformer 或 estimator 的别名, 第二个元素是 transformer 或 estimator
# 管道中的前两步是 transformer, 最后一步是 estimator
pipe_lr = Pipeline([
    ("scl", StandardScaler()),
    ("pca", PCA(n_components=2)),
    ("clf", LogisticRegression(random_state=1))
])
pipe_lr.fit(X_train, y_train)
print("Test Accuracy: %.3f" % pipe_lr.score(X_test, y_test))

# 分层 k 折交叉验证
skf = StratifiedKFold(n_splits=10)
kfold = skf.split(X_train, y=y_train)
# 手动计算精度
scores = []
for k, (train, test) in enumerate(kfold):
    pipe_lr.fit(X_train[train], y_train[train])
    score = pipe_lr.score(X_train[test], y_train[test])
    scores.append(score)
    print('Fold: %s, Class dist.: %s, Acc: %.3f' % (k + 1, np.bincount(y_train[train]), score))
print("CV accuracy (By hand): %.3f +/- %.3f" % (np.mean(scores), np.std(scores)))

# 利用 sklearn 自带的工具进行精度验证
# n_jobs 说明可以并行地进行 k 折评估, 充分利用多 CPU 的性能. 亲测, 不知道为何, n_jobs=1 用时反而是最短的
scores2 =cross_val_score(estimator=pipe_lr, X=X_train, y=y_train, cv=10, n_jobs=1)
print("CV accuracy (Using sklearn): %s" % scores2)
print("CV accuracy (Using sklearn): %.3f +/- %.3f" % (np.mean(scores2), np.std(scores2)))

