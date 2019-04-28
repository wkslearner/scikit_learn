#!/usr/bin/python
# encoding=utf-8

from numpy import *
import time
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris


iris=load_iris()
data=iris.data
data=data[:20,:]


'''欧式距离计算'''
def euclDistance(vector1, vector2):
    diff=vector2 - vector1   #数组差运算
    sqrt_diff=sum(power(diff,2))  #对结果求平方和

    return sqrt(sqrt_diff)  #返回马氏距离值


'''随机抽取k个样本'''
def initCentroids(dataSet, k):
    numSamples, dim = dataSet.shape  #求行列值
    centroids = zeros((k, dim))
    for i in range(k):
        index = int(random.uniform(0, numSamples))
        centroids[i, :] = dataSet[index, :]
    return centroids

arr=array([[1,2,3],[2,1,2],[3,3,2]])
print(nonzero(arr))

# cluster_algorithm cluster
def kmeans(dataSet, k):
    numSamples = dataSet.shape[0]  #计算数组的行数
    # first column stores which cluster this sample belongs to,
    # second column stores the error between this sample and its centroid
    clusterAssment = mat(zeros((numSamples, 2))) #构造矩阵
    clusterChanged = True

    ## step 1: 随机抽取k个样本作为中心点
    centroids = initCentroids(dataSet, k)

    while clusterChanged:
        clusterChanged = False
        # 循环样本
        for i in range(numSamples):
            minDist = 100000.0
            minIndex = 0
            # 循环每个中心点
            # step 2: 求每个中心点与第i个样本距离，找到最小距离的点
            for j in range(k):
                distance = euclDistance(centroids[j, :], dataSet[i, :])
                if distance < minDist:
                    minDist = distance
                    minIndex = j

                    ## step 3: update its cluster
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
                clusterAssment[i, :] = minIndex, minDist ** 2

        ## step 4: update centroids

        for j in range(k):
            pointsInCluster = dataSet[nonzero(clusterAssment[:, 0].A == j)[0]] #求每个簇所有点的属性值
            centroids[j, :] = mean(pointsInCluster, axis=0) #计算簇中点的所有属性均值，然后把它作为一个中心点

    print ('Congratulations, cluster complete!')
    return centroids, clusterAssment

centroids,cluster=kmeans(data,3)



# show your cluster only available with 2-D data
def showCluster(dataSet, k, centroids, clusterAssment):
    numSamples, dim = dataSet.shape
    if dim != 2:
        print ("Sorry! I can not draw because the dimension of your data is not 2!")
        return 1

    mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
    if k > len(mark):
        print ("Sorry! Your k is too large! please contact Zouxy")
        return 1

        # draw all samples
    for i in range(numSamples):
        markIndex = int(clusterAssment[i, 0])
        plt.plot(dataSet[i, 0], dataSet[i, 1], mark[markIndex])

    mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']
    # draw the centroids
    for i in range(k):
        plt.plot(centroids[i, 0], centroids[i, 1], mark[i], markersize=12)

    plt.show()


#showCluster(data,3,centroids,cluster)

''' 
from numpy import *
import time
import matplotlib.pyplot as plt

## step 1: load data  
print
"step 1: load data..."
dataSet = []
fileIn = open('E:/Python/Machine Learning in Action/testSet.txt')
for line in fileIn.readlines():
    lineArr = line.strip().split('\t')
    dataSet.append([float(lineArr[0]), float(lineArr[1])])

## step 2: clustering...  
print
"step 2: clustering..."
dataSet = mat(dataSet)
k = 4
centroids, clusterAssment = kmeans(dataSet, k)

## step 3: show the result  
print
"step 3: show the result..."
showCluster(dataSet, k, centroids, clusterAssment)
'''