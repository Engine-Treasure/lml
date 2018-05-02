# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

df = pd.read_csv("housing.data", header=None, sep="\s+")
df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS',
              'NOX', 'RM', 'AGE', 'DIS', 'RAD',
              'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

sns.set(style="whitegrid", context="notebook")
cols = ["LSTAT", "INDUS", "NOX", "RM", "MEDV"]
sns.pairplot(df[cols], size=2.5)  # 五个特征, 两两构造散点图矩阵
plt.show()


cm = np.corrcoef(df[cols].values.T)  # correlation coeffients
sns.set(font_scale=1.5)
hm = sns.heatmap(cm,
                 cbar=True,
                 annot=True,
                 square=True,
                 fmt=".2f",
                 annot_kws={"size": 15},
                 yticklabels=cols,
                 xticklabels=cols)
plt.show()
