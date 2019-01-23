#!/usr/bin/python
#coding:utf-8
#Author：Charlotte
import pandas as pd
import numpy as np
import time
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.cluster  import KMeans
from sklearn import metrics


class AutoEncoder():
    """ Auto Encoder
    layer      1     2    ...    ...    L-1    L
      W        0     1    ...    ...    L-2
      B        0     1    ...    ...    L-2
      Z              0     1     ...    L-3    L-2
      A              0     1     ...    L-3    L-2
    """

    def __init__(self, X, Y, nNodes):
        # training samples
        self.X = X
        self.Y = Y
        # 样本数量
        self.M = len(self.X)
        # 神经网络层数
        self.nLayers = len(nNodes)
        # 每个层级，神经元数量
        self.nNodes = nNodes
        # 网络参数(权重+偏置)
        self.W = list()
        self.B = list()
        self.dW = list()
        self.dB = list()
        self.A = list()
        self.Z = list()
        self.delta = list()
        for iLayer in range(self.nLayers - 1):
            #随机初始化权重
            self.W.append(
                np.random.rand(nNodes[iLayer] * nNodes[iLayer + 1]).reshape(nNodes[iLayer], nNodes[iLayer + 1]))
            #初始化偏置
            self.B.append(np.random.rand(nNodes[iLayer + 1]))
            self.dW.append(np.zeros([nNodes[iLayer], nNodes[iLayer + 1]]))
            self.dB.append(np.zeros(nNodes[iLayer + 1]))
            self.A.append(np.zeros(nNodes[iLayer + 1]))
            self.Z.append(np.zeros(nNodes[iLayer + 1]))
            self.delta.append(np.zeros(nNodes[iLayer + 1]))

        # value of cost function
        self.Jw = 0.0
        # active function (logistic function)
        self.sigmod = lambda z: 1.0 / (1.0 + np.exp(-z))
        # learning rate 1.2
        self.alpha = 2.5
        # steps of iteration 30000
        self.steps = 500

    '''反向传播'''
    def BackPropAlgorithm(self):
        # clear values
        self.Jw -= self.Jw
        for iLayer in range(self.nLayers - 1):
            self.dW[iLayer] -= self.dW[iLayer]
            self.dB[iLayer] -= self.dB[iLayer]
        # propagation (iteration over M samples)
        for i in range(self.M):
            # Forward propagation
            for iLayer in range(self.nLayers - 1):
                if iLayer == 0:  # first layer
                    self.Z[iLayer] = np.dot(self.X[i], self.W[iLayer])
                else:
                    self.Z[iLayer] = np.dot(self.A[iLayer - 1], self.W[iLayer])
                self.A[iLayer] = self.sigmod(self.Z[iLayer] + self.B[iLayer])
                # Back propagation
            for iLayer in range(self.nLayers - 1)[::-1]:  # reserve
                if iLayer == self.nLayers - 2:  # last layer
                    self.delta[iLayer] = -(self.X[i] - self.A[iLayer]) * (self.A[iLayer] * (1 - self.A[iLayer]))
                    self.Jw += np.dot(self.Y[i] - self.A[iLayer], self.Y[i] - self.A[iLayer]) / self.M
                else:
                    self.delta[iLayer] = np.dot(self.W[iLayer].T, self.delta[iLayer + 1]) * (
                                self.A[iLayer] * (1 - self.A[iLayer]))
                # calculate dW and dB
                if iLayer == 0:
                    self.dW[iLayer] += self.X[i][:, np.newaxis] * self.delta[iLayer][:, np.newaxis].T
                else:
                    self.dW[iLayer] += self.A[iLayer - 1][:, np.newaxis] * self.delta[iLayer][:, np.newaxis].T
                self.dB[iLayer] += self.delta[iLayer]
                # update
        for iLayer in range(self.nLayers - 1):
            self.W[iLayer] -= (self.alpha / self.M) * self.dW[iLayer]
            self.B[iLayer] -= (self.alpha / self.M) * self.dB[iLayer]


    '''执行自编码过程'''
    def PlainAutoEncoder(self):
        for i in range(self.steps):
            self.BackPropAlgorithm()
            #print ("step:%d" % i, "Jw=%f" % self.Jw)


    '''有效自编码'''
    def ValidateAutoEncoder(self):

        result_list=[]
        for i in range(self.M):
            #print (self.X[i])
            for iLayer in range(self.nLayers - 1):
                if iLayer == 0:  # input layer
                    self.Z[iLayer] = np.dot(self.X[i], self.W[iLayer])
                else:
                    self.Z[iLayer] = np.dot(self.A[iLayer - 1], self.W[iLayer])
                self.A[iLayer] = self.sigmod(self.Z[iLayer] + self.B[iLayer])

                if iLayer==0:
                    result_list.append(self.A[iLayer])

                #print ("\t layer=%d" % iLayer, self.A[iLayer])

        return  np.array(result_list)


iris=load_iris()

# # 归一化处理
# xx = preprocessing.scale(x)

#节点参数，网络层数和每层网络神经元数量
nNodes = np.array([4, 3, 4])
ae3 = AutoEncoder(iris.data, iris.data, nNodes)
ae3.PlainAutoEncoder()

#输入层到隐藏层编码
encoder_result=ae3.ValidateAutoEncoder()
print(encoder_result)

#加载数据
num_clusters = 3

clf = KMeans(n_clusters=num_clusters,  n_init=1, n_jobs = -1,verbose=1)
clf.fit(encoder_result)

print(clf.labels_)
labels = clf.labels_

#score是轮廓系数
score = metrics.silhouette_score(encoder_result, labels)

# clf.inertia_用来评估簇的个数是否合适，距离越小说明簇分的越好
print (clf.inertia_)
print (score)







