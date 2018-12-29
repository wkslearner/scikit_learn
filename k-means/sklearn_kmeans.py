
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from matplotlib import pyplot as plt
import warnings; warnings.simplefilter('ignore')  # Fix NumPy issues.
from sklearn.metrics import roc_curve,auc,roc_auc_score
from sklearn.preprocessing import StandardScaler
from plot_function.cluster_plot import plot_cluster_kmeans


iris=load_iris()
data=iris.data
target=iris.target

train_data=StandardScaler().fit_transform(data)

km=KMeans(n_clusters=4)
km.fit(train_data)

labels=km.labels_
center=km.cluster_centers_

print(km.inertia_)

plot_cluster_kmeans(data,labels)













