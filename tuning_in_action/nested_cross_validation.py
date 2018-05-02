# -*- coding: utf-8 -*-

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import learning_curve, validation_curve, cross_val_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_curve, auc, roc_auc_score, accuracy_score
from scipy import interp
from sklearn.linear_model import LogisticRegression

le = LabelEncoder()

df = pd.read_csv("wdbc.data", header=None)
X = df.loc[:, 2:].values
y = df.loc[:, 1].values
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

pipe_svc = Pipeline([
    ("scl", StandardScaler()),
    ("clf", SVC(random_state=1))
])

# 混淆矩阵
pipe_svc.fit(X_train, y_train)
y_pred = pipe_svc.predict(X_test)
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
print(confmat)

fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va="center", ha="center")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# 查准率, 查全率, f1
pre_score = precision_score(y_true=y_test, y_pred=y_pred)
rec_score = recall_score(y_true=y_test, y_pred=y_pred)
F1_score = f1_score(y_true=y_test, y_pred=y_pred)
print("Precision: %.3f" % pre_score)
print("Recall: %.3f" % rec_score)
print("F1: %.3f" % F1_score)

# ROC
X_train2 = X_train[:, [4, 14]]  # 从数据集中取 2 个特征构成新的特征集

pipe_lr = Pipeline([
    ("scl", StandardScaler()),
    ("clf", LogisticRegression(random_state=1))
])

skf = StratifiedKFold(n_splits=3, random_state=1)
cv = skf.split(X=X_train, y=y_train)
fig = plt.figure(figsize=(7, 5))

mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)  # 均匀地得到 0 到 1 之间 100 个点.
all_tpr = []
for i, (train, test) in enumerate(cv):
    # 基于 k 折交叉的方式, 计算 ROC

    # 用两个特征的训练集来训练对率回归模型, 并计算概率
    probas = pipe_lr.fit(X_train2[train], y_train[train]).predict_proba(X_train2[test])

    # 计算 ROC 性能指标
    fpr, tpr, thresholds = roc_curve(y_train[test], probas[:, 1], pos_label=1)
    roc_auc = auc(fpr, tpr)  # 计算 AUC
    plt.plot(fpr, tpr, lw=1, label="ROC fold %d (area = %0.2f)" % (i+1, roc_auc))

    mean_tpr += interp(mean_fpr, fpr, tpr)  # 线性插值
    mean_tpr[0] = 0.0

plt.plot([0, 1], [0, 1], linestyle="--", color=(0.6, 0.6, 0.6), label="random guessing")
plt.plot([0, 0, 1], [0, 1, 1], lw=2, linestyle=":", color="black", label="Perfect Performance")

mean_tpr /= 3
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, "k--", label="mean ROC (area = %0.2f)" % mean_auc, lw=2)

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.title('Receiver Operator Characteristic')
plt.legend(loc="lower right")
plt.show()


# ROC AUC Score
pipe_svc2 = pipe_svc.fit(X_train2, y_train)
y_pred2 = pipe_svc2.predict(X_test[:, [4, 14]])
print("ROC AUC: %.3f" % roc_auc_score(y_true=y_test, y_score=y_pred2))
print("Accuracy: %.3f" % accuracy_score(y_true=y_test, y_pred=y_pred2))

exit()

param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
param_grid = [
    {
        "clf__C": param_range,
        "clf__kernel": ["linear"]
    },
    {
        "clf__C": param_range,
        "clf__gamma": param_range,
        "clf__kernel": ["rbf"]
    }
]

# 这就实现了嵌套交叉验证了!
# 网格搜索是嵌套的内循环, 通过穷举搜素, 得到一个最优的模型
gs = GridSearchCV(estimator=pipe_svc,
                  param_grid=param_grid,
                  scoring="accuracy",
                  cv=10,
                  n_jobs=-1)
# 交叉验证的函数是外循环, 划分数据集, 用于内循环的模型选择; 该函数还有一个估计的过程, 在内循环完成模型选择之后, 用测试集进行性能评估
scores = cross_val_score(gs, X, y, scoring="accuracy", cv=5)
print("CV accuracy: %3f +/- %.3f" % (np.mean(scores), np.std(scores)))


gs = GridSearchCV(
    estimator=DecisionTreeClassifier(random_state=0),
    param_grid=[
        {"max_depth": [1, 2, 3, 4, 5, 6, 7, None]}
    ],
    scoring="accuracy",
    cv=5
)
scores = cross_val_score(gs, X, y, scoring="accuracy", cv=5)
print("CV accuracy: %3f +/- %.3f" % (np.mean(scores), np.std(scores)))
