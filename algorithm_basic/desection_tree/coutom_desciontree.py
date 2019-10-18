#!/usr/bin/python
# encoding=utf-8

import pandas as pd
import numpy as np
from  math import log
from numpy import *
from sklearn.datasets import load_iris


'''计算信息熵'''
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}

    #统计每个标签的数量
    for featVec in dataSet:
        currentLabel = featVec[-1]  #取出标签
        #往字典中添加标签
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1  #统计数据集中每个标签数量


    #计算信息熵
    shannonEnt = 0.0
    for key in labelCounts:
        #计算每类标签概率
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob*log(prob)

    return shannonEnt



'''分割数据集'''
def splitDataSet(dataSet, axis, value):
    #axis为数据维度(特征)，value每一列去重后的数值（特征值）
    retDataSet = []
    for featVec in dataSet:
        print(featVec)
        if featVec[axis] == value:
            print(featVec)
            reducedFeatVec = featVec[:axis]
            print(reducedFeatVec)
            reducedFeatVec.extend(featVec[axis+1:])
            print('xx',reducedFeatVec)
            retDataSet.append(reducedFeatVec)

    return retDataSet



'''根据信息增益，选择最好的分割点'''
def chooseBestFeatureToSplit(dataSet):  #默认最后一个类别为目标变量
    numFeatures = len(dataSet[0])-1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0; bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)  #对每一列数值进行去重操作
        newEntropy = 0.0

        #对数据进行分割
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i , value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy +=prob * calcShannonEnt(subDataSet) #计算子集信息熵
        infoGain = baseEntropy - newEntropy  #计算信息增益
        if(infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i

    return bestFeature



'''寻找频繁项集'''
def majorityCnt(classList):
    classCount = {}
    #统计每个类别数量
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1

    #按照类别倒序排序
    sortedClassCount = sorted(classCount.items(), reverse=True)

    return sortedClassCount[0][0]



'''决策树主函数'''
def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]  #提取类别变

    #判断是否只存在一个类别
    if classList.count(classList[0]) == len(classList):
        return classList[0]

    #如果数据是一维的情况下，遍历所有属性，选择频繁项
    # traversal all the features and choose the most frequent feature
    if (len(dataSet[0]) == 1):
        return majorityCnt(classList)

    #选择最佳分割点
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}

    del(labels[bestFeat])   #删除内存中的labels[bestFeat]
    #get the list which attain the whole properties
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)

    #对树进行递归调用，分割子节点
    for value  in uniqueVals:
        subLabels = labels[:]
        #对子集进行决策树过程，这里sublabels取法需要注意
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)

    return myTree




def classify(inputTree, featLabels, testVec):
    #输入的树模型
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else: classLabel = secondDict[key]
    return classLabel




if __name__=='__main__':

    iris=load_iris()
    data=iris.data
    col=iris.feature_names
    target=iris.target

    iris_df=pd.DataFrame(data,columns=['col_1','col_2','col_3','col_4'])
    iris_df['target']=target
    iris_df=iris_df[iris_df['target']<2]


    # myTree = createTree(np.array(iris_df[['col_1','col_2','col_3','col_4']]), np.array(iris_df['target']))
    # print (myTree)


    '''创建数据集'''
    def createDataSet():
        dataSet = [[1,1,'yes'],
                   [1,1, 'yes'],
                   [1,0,'no'],
                   [0,1,'no'],
                   [0,1,'no'],
                   [2,1,'yes'],
                   [0,2,'no'],
                   [2,2,'yes']]
        labels = ['no surfacing','flippers']
        return dataSet, labels

    dataset,labels=createDataSet()
    xlabels=labels

    myTree = createTree(dataset,xlabels)
    print(myTree)