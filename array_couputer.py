#!/usr/bin/python
# encoding=utf-8

import pandas as pd
import numpy as np
from data_process.list_process import remove_list
from data_process.feature_handle import disper_split
from sklearn import preprocessing
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV, KFold ,RandomizedSearchCV
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import make_scorer, fbeta_score, accuracy_score,recall_score
from sklearn import metrics
import time
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from scipy import sparse
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils import check_array, check_consistent_length
from sklearn.utils.testing import ignore_warnings
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster._dbscan_inner import dbscan_inner
import random
from sklearn.datasets.samples_generator import make_blobs
import Levenshtein



'''获取用户下单行为'''
def get_order_behave(start_time='',end_time=''):

    start_mon=pd.Period(start_time,freq='M')
    end_mon=pd.Period(end_time,freq='M')
    num=end_mon-start_mon
    print(start_mon)

    mon_list=[]
    for i in range(num):
        this_mon=start_mon+i
        this_mon=str(this_mon).replace('-','')
        mon_list.append(this_mon)

    print(mon_list)



# get_order_behave(start_time='2019-01-01',end_time='2019-05-01')



# center = [[1, 1], [-1, -1], [1, -1]]
# cluster_std = 0.35
# X1, Y1 = make_blobs(n_samples=300, centers=center,
#                     n_features=2, cluster_std=cluster_std, random_state=1)

# print(X1)
# dbscan_cluster=DBSCAN(eps=0.4)
# dbscan_cluster.fit(X1)

# print(X1)
# print(dbscan_cluster.labels_)
# pre=dbscan_cluster.fit_predict([[-1.16638051 -0.83283644],
#                                 [-1.0421626  -1.43159226]])


# km=KMeans(n_clusters=3)
# km.fit(X1)
# print(X1)
# print(km.labels_)
# pre=km.predict([[-1.16638051,-0.83283644]])
# print(pre)


a=[['1','2','3'],['4','5','6']]
b=['2','3','4']

sim=Levenshtein.seqratio(a,b)
print(sim)



#!/usr/bin/python
# coding=utf-8
from numpy import *
# 加载数据
def loadDataSet(fileName):  # 解析文件，按tab分割字段，得到一个浮点数字类型的矩阵
    dataMat = []              # 文件的最后一个字段是类别标签
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = map(float, curLine)    # 将每个元素转成float类型
        dataMat.append(fltLine)
    return dataMat

# 计算欧几里得距离
def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2))) # 求两个向量之间的距离


# 构建聚簇中心，取k个(此例中为4)随机质心
def randCent(dataSet, k):
    n = shape(dataSet)[1]
    centroids = mat(zeros((k,n)))   # 每个质心有n个坐标值，总共要k个质心
    for j in range(n):
        minJ = min(dataSet[:,j])
        maxJ = max(dataSet[:,j])
        rangeJ = float(maxJ - minJ)
        centroids[:,j] = minJ + rangeJ * random.rand(k, 1)
    return centroids


# k-means 聚类算法
def kMeans(dataSet, k, distMeans =distEclud, createCent = randCent):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m,2)))    # 用于存放该样本属于哪类及质心距离
    # clusterAssment第一列存放该数据所属的中心点，第二列是该数据到中心点的距离
    centroids = createCent(dataSet, k)
    clusterChanged = True   # 用来判断聚类是否已经收敛
    while clusterChanged:
        clusterChanged = False;
        for i in range(m):  # 把每一个数据点划分到离它最近的中心点
            minDist = inf; minIndex = -1;
            for j in range(k):
                distJI = distMeans(centroids[j,:], dataSet[i,:])
                if distJI < minDist:
                    minDist = distJI; minIndex = j  # 如果第i个数据点到第j个中心点更近，则将i归属为j
            if clusterAssment[i,0] != minIndex:
                clusterChanged = True; # 如果分配发生变化，则需要继续迭代
            clusterAssment[i,:] = minIndex,minDist**2   # 并将第i个数据点的分配情况存入字典
        print (centroids)
        for cent in range(k):   # 重新计算中心点
            ptsInClust = dataSet[nonzero(clusterAssment[:,0].A == cent)[0]]   # 去第一列等于cent的所有列
            centroids[cent,:] = mean(ptsInClust, axis = 0)  # 算出这些数据的中心点

    return centroids, clusterAssment
# --------------------测试----------------------------------------------------
# 用测试数据及测试kmeans算法
datMat = mat(loadDataSet('testSet.txt'))
myCentroids,clustAssing = kMeans(datMat,4)
print (myCentroids)
print (clustAssing)