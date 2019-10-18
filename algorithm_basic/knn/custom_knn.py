#!/usr/bin/python
# encoding=utf-8

from numpy import *
import operator
from  sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split

iris=load_iris()
data=iris.data
target=iris.target

x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2)


# classify using kNN
def knnclassify(newInput, dataSet, labels, k):

    numSamples = dataSet.shape[0]  # shape[0] stands for the num of row

    ## step 1: 计算距离
    # tile(A, reps): 对数组A重复，重组成dataset的形式，下面例子是变成4行1列
    # the following copy numSamples rows for dataSet
    diff = tile(newInput, (numSamples, 1)) - dataSet  # 计算数组差值
    squaredDiff = diff ** 2
    squaredDist = sum(squaredDiff, axis=1)  # 按列求和
    distance = squaredDist ** 0.5  #对距离平方和开根号


    ## step 2: 对距离进行排序
    # argsort() 返回距离排序后的索引
    sortedDistIndices = argsort(distance)


    '''求最近k个样本值'''
    k_sample={}
    for i in range(len(sortedDistIndices)):
        if sortedDistIndices[i]<k:
            sam=dataSet[i]
            k_sample[sortedDistIndices[i]]=sam


    classCount = {}
    for i in range(k):
        ## step 3: 选择最新k个距离的样本标签
        voteLabel = labels[sortedDistIndices[i]]

        ## step 4: 计算每个标签出现的次数
        # when the key voteLabel is not in dictionary classCount, get()
        # will return 0
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1


        ## step 5: 返回出现次数最大的标签结果
    maxCount = 0
    for key, value in classCount.items():
        if value > maxCount:
            maxCount = value
            maxIndex = key

    return maxIndex,k_sample


#dataSet, labels = createDataSet()
sample=[[ 2.1 , 3.2 , 1.4 , 2.2]]
category,k_sam=knnclassify(sample,data,target,4)

print(category,k_sam)




'''
testX = array([1.2, 1.0])
k = 3
outputLabel = kNNClassify(testX, dataSet, labels, 3)
print ("Your input is:", testX, "and classified to class: ", outputLabel)


testX = array([0.1, 0.3])
outputLabel = kNNClassify(testX, dataSet, labels, 3)
print ("Your input is:", testX, "and classified to class: ", outputLabel )
'''