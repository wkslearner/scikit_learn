import pandas as pd
import urllib
import numpy as np
from itertools import combinations
import types
import numbers


'''基尼系数计算'''
def Gini(df, y, var, w):
    '''
    :param df: 数据集，包含了需要计算Gini指数的特征
    :param y: 标签，0或1
    :param w: 样本权重
    :param var: 根据var将df切分为有限的几个子集，再根据权重计算相应的Gini
    :return: Gini指数
    '''

    #先定义计算数据集的Gini函数，再计算基于属性var的Gini指数
    def GiniDataSet(df,y,w):
        p1 = sum(df.apply(lambda x: x[y]*x[w],axis = 1))/sum(df[w])
        p2 = 1-p1
        return 1- p1**2-p2**2

    gini = 0
    for v in set(df[var]):
        temp = df.loc[df[var] == v]
        gini += sum(temp[w]) * GiniDataSet(temp,y,w)

    return gini


'''基尼系数计算'''
def Gini2(df, y, var):
    '''
    :param df: 数据集，包含了需要计算Gini指数的特征
    :param y: 标签，0或1
    :param w: 样本权重
    :param var: 根据var将df切分为有限的几个子集，再根据权重计算相应的Gini
    :return: Gini指数
    '''

    #先定义计算数据集的Gini函数，再计算基于属性var的Gini指数
    def GiniDataSet2(df,y):
        p1 = sum(df[y])*1.0/len(df[y])
        p2 = 1 - p1
        return 1- p1**2-p2**2

    gini = 0
    for v in set(df[var]):
        temp = df.loc[df[var] == v]
        gini += temp.shape[0]* GiniDataSet2(temp,y)/df.shape[0]

    return gini


'''特殊情况下的分类特征处理'''
def Terminate(df, feature_list, y, w):
    '''
    :param df: 当前构建CART的数据集
    :param feature_list: 当前构建CART的特征集
    :param y: 标签
    :param w: 样本权重
    :return: 终止与否
    '''
    w_sum = sum(df[w])
    majorityClass = int(sum(df.apply(lambda x: x[y]*x[w]/w_sum,axis=1))>=0.5) #判断权重是否大于0.5

    if df.shape[0]<=5:  #数据的行数低于5行
        return majorityClass
    if feature_list == []: #变量列表为空
        return majorityClass
    if df[feature_list].drop_duplicates().shape[0] == 1:  #去重后的行数为1
        return majorityClass
    if len(set(df[y])) == 1:  #变量y只有一个取值的情况下
        return majorityClass

    return 'subTree'



'''判断变量类型'''
def FeatureType(df, var):
    if isinstance(df.iloc[0][var],numbers.Real):
        return 'numerical'
    else:
        return 'categorical'


'''数值型变量分割'''
def NumFeatureSplit(df,var,y, w):
    '''
    :param df:
    :param var:
    :param y:
    :param w:样本权重
    :return:
    '''
    x = sorted(list(set(df[var])))
    split_gini = {}
    N = df.shape[0]
    for i in range(1,len(x)):
        threshold = 0.5*(x[i-1]+x[i])
        df['separate_group'] = df.apply(lambda x: int(x[var]<=threshold), axis=1)
        varGini = Gini(df, y, 'separate_group', w)
        split_gini[threshold] = varGini
    del df['separate_group']
    return {'splittedPoint' : min(split_gini,key=split_gini.get), 'minGini' : min(split_gini.values())}


'''离散型变量分割'''
def CatFeatureSplit(df,var,y,w):
    '''
    :param df:
    :param var:
    :param y:
    :param w:样本权重
    :return:
    '''
    avgLabel = df.groupby([var])[y].mean().to_frame()
    avgLabelSorted = avgLabel.sort_values(by=y)
    CatVar = list(avgLabelSorted.index)
    Cat2Num = {CatVar[i]:i for i in range(len(CatVar))}
    df['temp_col'] = df[var].map(Cat2Num)
    splitNumerical = NumFeatureSplit(df, 'temp_col', y, w)
    splitted_new_var = [k for k in Cat2Num.keys() if Cat2Num[k]<=splitNumerical['splittedPoint']]
    return {'splittedPoint':splitted_new_var,'minGini':splitNumerical['minGini']}


'''cart训练函数'''
def TrainCART(df,feature_list,y, w, depth = 10000):
    '''
    :param df: 建立CART的数据集
    :param feature_list: 用于模型开发的特征的列表
    :param y: 标签，取值0，1
    :param w: 样本权重。
    :param depth: 树的深度。当指明深度时，设置该参数。
    :return: 叶子节点，或者子树。最终返回字典形式的CART树
    '''

    child = Terminate(df, feature_list, y,w)

    w_sum = sum(df[w])
    majorityClass = int(sum(df.apply(lambda x: x[y]*x[w]/w_sum,axis=1))>=0.5)
    if child in [0,1]:  #判断是否终止循环
        return child
    if depth == 0:
        return majorityClass

    #变量基尼系数计算
    featureGini = {}
    for feature in feature_list:
        if len(set(df[feature])) == 1:
            if len(feature_list) == 1:
               return majorityClass   #判断是否只存在一个变量或者变量只存在一个值
            else:
                feature_list.remove(feature)
                continue
        if FeatureType(df,feature) == 'numerical':  #判断数据取值类型
            featureSplit = NumFeatureSplit(df, feature, y, w)
        else:
            featureSplit = CatFeatureSplit(df, feature, y, w)
        featureGini[feature] = [featureSplit['splittedPoint'],featureSplit['minGini']]  #返回分割点和分割基尼系数
    sortedFeatureGini = sorted(featureGini.items(),key=lambda x: x[1][1]) #根据基尼系数进行排序
    bestFeature, bestSplit = sortedFeatureGini[0][0], sortedFeatureGini[0][1][0]  #返回基尼最小的变量和最佳分割点
    print(bestFeature,bestSplit)

    cartResult = {bestFeature:{}}
    if  bestSplit.__class__ == list: #判断是否是分类变量
        subTreeFeatures = [i for i in feature_list if i != bestFeature] #剔除上一轮最佳分割属性后，剩下的属性
        bestFeaturesVals = list(set(df[bestFeature]))

        leftTree = df.loc[df[bestFeature].isin(bestSplit)] #左边树
        leftVals = bestSplit

        cartResult[bestFeature][str(leftVals)] = TrainCART(leftTree,subTreeFeatures,y, w, depth-1) #递归调用

        rightTree = df.loc[~df[bestFeature].isin(bestSplit)] #右边数
        rightVals = [i for i in bestFeaturesVals if i not in set(leftVals)]
        cartResult[bestFeature][str(rightVals)] = TrainCART(rightTree, subTreeFeatures, y, w, depth-1)
    else:
        subTreeFeatures = [i for i in feature_list if i != bestFeature]
        leftTree = df.loc[df[bestFeature]<=bestSplit]
        cartResult[bestFeature]["<="+str(bestSplit)] = TrainCART(leftTree, subTreeFeatures, y, w, depth-1)
        rightTree = df.loc[df[bestFeature]>bestSplit]
        cartResult[bestFeature][">"+str(bestSplit)] = TrainCART(rightTree, subTreeFeatures, y, w, depth-1)

    return cartResult


'''树结果预测'''
def predCART(record, cart):
    '''
    :param record: 
    :param cart: 
    :return: 
    '''
    root, subTree = cart.items()[0]
    for k, v in subTree.items():
        if type(v).__name__ == 'dict':
            return predCART(record, v)
        else:
            if k.find('>') > -1 or k.find('<=') > -1:
                if eval(str(record[root])+k):
                    return v
                else:
                    return 1-v
            else:
                if record[root] in eval(k):
                    return v
                else:
                    return 1-v



def CalcDictDepth(dict):
    v_list = dict.values()[0]
    v_left = v_list.values()[0]
    v_right = v_list.values()[1]
    if type(v_left).__name__ != 'dict' and type(v_right).__name__ != 'dict':
        return 1
    else:
        if type(v_left).__name__ == 'dict' and type(v_right).__name__ != 'dict':
            return 1+CalcDictDepth(v_left)
        elif type(v_left).__name__ != 'dict' and type(v_right).__name__ == 'dict':
            return 1 + CalcDictDepth(v_right)
        else:
            return max(1+CalcDictDepth(v_right), 1+CalcDictDepth(v_left))