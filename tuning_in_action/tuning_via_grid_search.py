# -*- coding: utf-8 -*-

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import learning_curve, validation_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

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

gs = GridSearchCV(estimator=pipe_svc,
                  param_grid=param_grid,
                  scoring="accuracy",
                  cv=10,
                  n_jobs=-1)
gs = gs.fit(X_train, y_train)
print(gs.best_score_)
print(gs.best_params_)
clf = gs.best_estimator_
clf.fit(X_train, y_train)
print("Test accuracy: %.3f" % clf.score(X_test, y_test))

