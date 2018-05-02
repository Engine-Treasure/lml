# -*- coding: utf-8 -*-

from matplotlib import cm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_samples
from sklearn.datasets import make_blobs, make_moons
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram

X, y = make_blobs(n_samples=150,
                  n_features=2,
                  centers=3,
                  cluster_std=0.5,
                  shuffle=True,
                  random_state=0)

km = KMeans(n_clusters=3,
            init="random",
            n_init=10,  # 独立执行 10 次算法, 最终将返回 SSE 最小的模型
            max_iter=300,  # 此处是最大迭代次数是单次执行 k 均值算法的最大迭代次数
            tol=1e-04,
            random_state=0)
y_km = km.fit_predict(X)
print(km.inertia_)  # 评估性能

plt.scatter(X[y_km == 0, 0], X[y_km == 0, 1], s=50, c="lightgreen", marker="s", label="cluster 1")
plt.scatter(X[y_km == 1, 0], X[y_km == 1, 1], s=50, c="orange", marker="o", label="cluster 2")
plt.scatter(X[y_km == 2, 0], X[y_km == 2, 1], s=50, c="lightblue", marker="v", label="cluster 3")
plt.scatter(km.cluster_centers_[:, 0],
            km.cluster_centers_[:, 1],
            s=250,
            marker="*",
            c="red",
            label="centroids")
plt.legend()
plt.grid()
plt.show()

# elbow
distortions = []
for i in range(1, 11):
    km = KMeans(n_clusters=i,
                init="k-means++",
                n_init=10,
                max_iter=300,
                random_state=0)
    km.fit(X)
    distortions.append(km.inertia_)
plt.plot(range(1, 11), distortions, marker="o")
plt.xlabel("Number of clusters")
plt.ylabel("Distortion")
plt.show()

# silhouette analysis
km = KMeans(n_clusters=3,
            init="k-means++",
            n_init=10,
            max_iter=300,
            tol=1e-04,
            random_state=0)
y_km = km.fit_predict(X)

cluster_labels = np.unique(y_km)
n_clusters = cluster_labels.shape[0]
# 为每个样本计算 silhouette 系数
silhouette_vals = silhouette_samples(X, y_km, metric="euclidean")
y_ax_lower, y_ax_upper = 0, 0  # 用于柱状图在 y 轴上的宽度
yticks = []

for i, c in enumerate(cluster_labels):
    c_silhouette_vals = silhouette_vals[y_km == c]  # 某一集群的 silhouette 系数
    c_silhouette_vals.sort()
    y_ax_upper += len(c_silhouette_vals)
    color = cm.jet(i / n_clusters)
    plt.barh(range(y_ax_lower, y_ax_upper),  # 就是 len(c_silouette_vals 的长度
             c_silhouette_vals,
             height=1.0,
             edgecolor="none",
             color=color)
    yticks.append((y_ax_lower + y_ax_upper) / 2)  # y 轴提示的位置
    y_ax_lower += len(c_silhouette_vals)
silhouette_avg = np.mean(silhouette_vals)
plt.axvline(silhouette_avg, color="red", linestyle="--")  # 绘制一条垂直于 x 轴的直线
plt.yticks(yticks, cluster_labels + 1)
plt.ylabel("Cluster")
plt.xlabel("Silhouette coefficient")
plt.show()

# bad clustering sample
km = KMeans(n_clusters=2,
            init="k-means++",
            max_iter=300,
            tol=1e-04,
            random_state=0)
y_km = km.fit_predict(X)
plt.scatter(X[y_km == 0, 0], X[y_km == 0, 1], s=50, c="lightgreen", marker="s", label="cluster 1")
plt.scatter(X[y_km == 1, 0], X[y_km == 1, 1], s=50, c="orange", marker="o", label="cluster 2")
plt.scatter(km.cluster_centers_[:, 0],
            km.cluster_centers_[:, 1],
            s=250,
            marker="*",
            c="red",
            label="centroids")
plt.legend()
plt.grid()
plt.show()

cluster_labels = np.unique(y_km)
n_clusters = cluster_labels.shape[0]
# 为每个样本计算 silhouette 系数
silhouette_vals = silhouette_samples(X, y_km, metric="euclidean")
y_ax_lower, y_ax_upper = 0, 0  # 用于柱状图在 y 轴上的宽度
yticks = []

for i, c in enumerate(cluster_labels):
    c_silhouette_vals = silhouette_vals[y_km == c]  # 某一集群的 silhouette 系数
    c_silhouette_vals.sort()
    y_ax_upper += len(c_silhouette_vals)
    color = cm.jet(i / n_clusters)
    plt.barh(range(y_ax_lower, y_ax_upper),  # 就是 len(c_silouette_vals 的长度
             c_silhouette_vals,
             height=1.0,
             edgecolor="none",
             color=color)
    yticks.append((y_ax_lower + y_ax_upper) / 2)  # y 轴提示的位置
    y_ax_lower += len(c_silhouette_vals)
silhouette_avg = np.mean(silhouette_vals)
plt.axvline(silhouette_avg, color="red", linestyle="--")  # 绘制一条垂直于 x 轴的直线
plt.yticks(yticks, cluster_labels + 1)
plt.ylabel("Cluster")
plt.xlabel("Silhouette coefficient")
plt.show()

# complete agglomerative hierarchical clustering

# 造数据
np.random.seed(123)
variables = ["X", "Y", "Z"]
labels = ["ID_0", "ID_1", "ID_2", "ID_3", "ID_4"]
X = np.random.random_sample([5, 3]) * 10  # 得到 5*3 的随机矩阵, 然后乘以标量 10
df = pd.DataFrame(X, columns=variables, index=labels)
print(df)

# 基于特征 X, Y, Z 对数据集中每一对样本计算欧式距离
# 得到的是一个压缩距离矩阵 condensed distance matrix
# row_dist = pd.DataFrame(squareform(pdist(df, metric="euclidean")), columns=labels, index=labels)

row_clusters = linkage(df.values,
                       metric="euclidean",
                       method="complete")
# df_row_clusters = pd.DataFrame(row_clusters,
#                                columns=["row label 1", "row label 2", "distance", "no. of items in clust."],
#                                index=["cluster %d" % (i + 1) for i in range(row_clusters.shape[0])])

# 系统树图
# 创建黑白的系统树图配色 1/2
# from scipy.cluster.hierarchy import set_link_color_palette
# set_link_color_palette(["black"])
row_dendr = dendrogram(row_clusters,
                       labels=labels,
                       # 创建黑白的系统树图配色 2/2
                       # color_threshold=np.inf
                       )
plt.tight_layout()
plt.ylabel("Euclidean distance")
plt.show()

# attaching dendrograms to heat map

# 造数据
np.random.seed(123)
variables = ["X", "Y", "Z"]
labels = ["ID_0", "ID_1", "ID_2", "ID_3", "ID_4"]
X = np.random.random_sample([5, 3]) * 10  # 得到 5*3 的随机矩阵, 然后乘以标量 10
df = pd.DataFrame(X, columns=variables, index=labels)
print(df)

# 基于特征 X, Y, Z 对数据集中每一对样本计算欧式距离
# 得到的是一个压缩距离矩阵 condensed distance matrix
# row_dist = pd.DataFrame(squareform(pdist(df, metric="euclidean")), columns=labels, index=labels)

row_clusters = linkage(df.values,
                       metric="euclidean",
                       method="complete")
# df_row_clusters = pd.DataFrame(row_clusters,
#                                columns=["row label 1", "row label 2", "distance", "no. of items in clust."],
#                                index=["cluster %d" % (i + 1) for i in range(row_clusters.shape[0])])

fig = plt.figure(figsize=(8, 8))
# 指定系统树图的 x axis position, y axis position, width, height
axd = fig.add_axes([0.09, 0.1, 0.2, 0.6])

# 系统树图
# 创建黑白的系统树图配色 1/2
# from scipy.cluster.hierarchy import set_link_color_palette
# set_link_color_palette(["black"])
row_dendr = dendrogram(row_clusters,
                       labels=labels,
                       orientation="left"
                       # 创建黑白的系统树图配色 2/2
                       # color_threshold=np.inf
                       )

# 记录数据
df_rowclust = df.ix[row_dendr["leaves"][::-1]]

# 从以上 DataFrame 中构建热图
axm = fig.add_axes([0.23, 0.1, 0.6, 0.6])
cax = axm.matshow(df_rowclust, interpolation="nearest", cmap="hot_r")

axd.set_xticks([])
axd.set_yticks([])
for i in axd.spines.values():
    i.set_visible(False)
fig.colorbar(cax)
axm.set_xticklabels([""] + list(df_rowclust.columns))
axm.set_yticklabels([""] + list(df_rowclust.index))
plt.ylabel("Euclidean distance")
plt.show()

# agglomerative clustering via sklearn

np.random.seed(123)
variables = ["X", "Y", "Z"]
labels = ["ID_0", "ID_1", "ID_2", "ID_3", "ID_4"]
X = np.random.random_sample([5, 3]) * 10  # 得到 5*3 的随机矩阵, 然后乘以标量 10

ac = AgglomerativeClustering(n_clusters=2,  # 用于指定要找到的集群数
                             affinity="euclidean",
                             linkage="complete")
labels = ac.fit_predict(X)
print("Cluster labels: %s" % labels)

# k-means vs Agglomerative clustering vs DBSCAN

X, y = make_moons(n_samples=200, noise=0.05, random_state=0)

f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 3))
km = KMeans(n_clusters=2, random_state=0)
y_km = km.fit_predict(X)
ax1.scatter(X[y_km == 0, 0], X[y_km == 0, 1], c="lightblue", marker="o", s=40, label="cluster 1")
ax1.scatter(X[y_km == 1, 0], X[y_km == 1, 1], c="red", marker="s", s=40, label="cluster 2")
ax1.set_title("K-means clustering")
ax1.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], s=250, marker="*", c="red", label="centroids")

ac = AgglomerativeClustering(n_clusters=2, affinity="euclidean", linkage="complete")
y_ac = ac.fit_predict(X)
ax2.scatter(X[y_ac == 0, 0], X[y_ac == 0, 1], c="lightblue", marker="o", s=40, label="cluster 1")
ax2.scatter(X[y_ac == 1, 0], X[y_ac == 1, 1], c="red", marker="s", s=40, label="cluster 2")
ax2.set_title("Agglomerative clustering")

db = DBSCAN(eps=0.2, min_samples=5, metric="euclidean")
y_db = db.fit_predict(X)
ax3.scatter(X[y_db == 0, 0], X[y_db == 0, 1], c="lightblue", marker="o", s=40, label="cluster 1")
ax3.scatter(X[y_db == 1, 0], X[y_db == 1, 1], c="red", marker="s", s=40, label="cluster 2")
ax3.set_title("DBSCAN")

plt.legend()
plt.show()
