
'''
Density-based spatial clustering of applications with noise
基于密度聚类算法（内含噪声）
'''

print(__doc__)

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
from numpy import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import DBSCAN
from plot_function.cluster_plot import plot_cluster_DBSCAN

iris=load_iris()
data=iris.data
target=iris.target

dataset=data.copy()

#数据标准化
data=StandardScaler().fit_transform(data)
print(data.shape[0])

#对数据进行DBSCAN聚类
cluster=DBSCAN()
cluster.fit(data)

#分类标签
labels=cluster.labels_
#分类噪声
noise=list(labels).count(-1)

#分类核心点索引
core_index=cluster.core_sample_indices_
print(noise)

labels_num=set(labels)

#生成类似标签数组(数据为布尔类型)
core_samples_mask = np.zeros_like(cluster.labels_, dtype=bool)
#用核心样本索引替换指定位置的值
core_samples_mask[cluster.core_sample_indices_] = True


plot_cluster_DBSCAN(dataset,labels,core_samples_mask)




'''
# Generate sample data 生成样本数据
centers = [[1, 1], [-1, -1], [1, -1]]
X, labels_true = make_blobs(n_samples=750, centers=centers, cluster_std=0.4,random_state=0)

# 数据标准化
X = StandardScaler().fit_transform(X)

# Compute DBSCAN
db = DBSCAN(eps=0.3, min_samples=10).fit(X)
print(db.core_sample_indices_.shape)


#生成类似标签数组(数据为布尔类型)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
#用核心样本索引替换指定位置的值
core_samples_mask[db.core_sample_indices_] = True
print(core_samples_mask.shape)

labels = db.labels_

#类别数量
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

#统计噪声的数量
n_noise_ = list(labels).count(-1)

# print('Estimated number of clusters: %d' % n_clusters_)
# print('Estimated number of noise points: %d' % n_noise_)
# print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
# print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
# print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
# print("Adjusted Rand Index: %0.3f"
#       % metrics.adjusted_rand_score(labels_true, labels))
# print("Adjusted Mutual Information: %0.3f"
#       % metrics.adjusted_mutual_info_score(labels_true, labels))
# print("Silhouette Coefficient: %0.3f"
#       % metrics.silhouette_score(X, labels))


# Black removed and is used for noise instead.
unique_labels = set(labels)
print(unique_labels)
print([each for each in np.linspace(0,1,len(unique_labels))])

#指定类别颜色
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

#按照不同的类别，对不同的点进行区分颜色
for k, col in zip(unique_labels, colors):
    if k == -1:
        # 把噪声涂上黑色
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)
    #指定特定类别的点(核心点)
    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=14)
    #指定类别（非核心点）
    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()

'''







