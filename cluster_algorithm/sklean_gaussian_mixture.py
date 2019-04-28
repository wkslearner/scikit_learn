
'''
高斯混合聚类
'''

print(__doc__)

from sklearn.mixture import GaussianMixture
from sklearn.datasets import load_iris
from plot_function.cluster_plot import plot_cluster
from sklearn.preprocessing import StandardScaler

iris=load_iris()
data=iris.data
target=iris.target

train_data=StandardScaler().fit_transform(data)

gm=GaussianMixture(n_components=4)
gm.fit(train_data)

labels=gm.predict(train_data)

plot_cluster(train_data,labels)

# labels=gm.labels_
# center=gm.cluster_centers_
# print(gm.inertia_)
# plot_cluster(data,labels)


