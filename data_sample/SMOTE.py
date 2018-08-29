#!/usr/bin/python
# encoding=utf-8

import random
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler



def disper_split(dataframe,var_list):
    '''
    :param dataframe: 目标数据框
    :param var_list: 分类变量列表
    :return: 变量与数值映射字典及分类处理后的新数据框
    '''
    split_point_cat={}
    split_cat_list = []
    for var in var_list:
        split_cat_list.append(var)
        mid_dict={}
        if dataframe[dataframe[var].isnull()].shape[0] > 0:
            sort_value = sorted(list(dataframe[dataframe[var].notnull()][var].unique()))
            num = len(sort_value)
            for i in range(num):
                dataframe.loc[(dataframe[var] == sort_value[i]), var ] = i
                mid_dict[i]=sort_value[i]

            dataframe.loc[dataframe[var].isnull(), var] = -1
            mid_dict[-1]='None'
            split_point_cat[var]=mid_dict

        else:
            sort_value = sorted(list(dataframe[dataframe[var].notnull()][var].unique()))
            num = len(sort_value)
            for i in range(num):
                dataframe.loc[(dataframe[var] == sort_value[i]), var] = i
                mid_dict[i] = sort_value[i]

            split_point_cat[var ] = mid_dict

    return  dataframe




'''SMOTE人工合成样本'''
class Smote:

    def __init__(self,samples,N=10,k=5):
        self.n_samples,self.n_attrs=samples.shape  #数据的行列
        self.N=N
        self.k=k
        self.samples=samples
        self.newindex=0
        #self.synthetic=np.zeros((self.n_samples*N,self.n_attrs))


    def over_sampling(self):
        N=int(self.N/100)
        self.synthetic = np.zeros((self.n_samples * N, self.n_attrs))  #建立与传入对象相同的0值数组
        #建立k邻近点模型
        neighbors=NearestNeighbors(n_neighbors=self.k).fit(self.samples)
        print ('neighbors',neighbors)
        for i in range(len(self.samples)):
            #对每个样本求其k个邻近点
            nnarray=neighbors.kneighbors(self.samples[i].reshape(1,-1),return_distance=False)[0]
            self._populate(N,i,nnarray)

        return self.synthetic


    # for each minority class samples,choose N of the k nearest neighbors and generate N synthetic samples.
    #对于每个少数类样本，选择k个最近邻的n，并生成n个合成样本。

    def _populate(self,N,i,nnarray):
        for j in range(N):
            #随机选取某一样本周围的nn个样本
            nn=random.randint(0,self.k-1)
            #计算这个样本到nn个点的距离
            dif=self.samples[nnarray[nn]]-self.samples[i]
            #生成一个随机数
            gap=random.random()
            #合成人工样本
            self.synthetic[self.newindex]=self.samples[i]+gap*dif
            self.newindex+=1



if __name__=='__main__':
    df = pd.read_excel('/Users/andpay/Documents/job/data/帮还活动/activity_history/marketing_modedata3_14.xlsx')
    df = df[0:100]

    cate_list = ['sex', 'brandcode', 'channel_type', 'marry', 'ccerate']
    df = disper_split(df, cate_list)
    df = df.fillna(0)
    var_list = list(df.columns)
    var_list.remove('partyid')
    var_list.remove('name')

    #a=np.array([[1,2,3],[4,5,6],[2,3,1],[2,1,2],[2,3,4],[2,3,4]])
    data=np.array(df[var_list])
    print(np.round(data,3))
    s=Smote(data,N=50)

    print (s.over_sampling())
    print (s.newindex)

