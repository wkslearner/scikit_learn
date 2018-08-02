#!/usr/bin/python
# encoding=utf-8

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

iris=load_iris()
data=iris.data
col=iris.feature_names
target=iris.target

iris_df=pd.DataFrame(data,columns=['col_1','col_2','col_3','col_4'])
iris_df['target']=target
iris_df=iris_df[iris_df['target']<2]
#print(iris_df)


'''基于信息增益的连续变量离散化'''
class Feature_Discretization(object):
    def __init__(self):
        self.min_interval = 1
        self.min_epos = 0.05
        self.final_bin = []


    def fit(self, x, y, min_interval=1):
        '''
        :param x: 需要离散化变量
        :param y: 目标变量
        :param min_interval: 最小间隔
        :return: 
        '''
        self.min_interval = min_interval
        x = np.floor(x)
        x = np.int32(x)
        min_val = np.min(x)
        bin_dict = {}
        bin_li = []

        #统计每个pos下面y的0，1类型的出现次数
        for i in range(len(x)):
            pos = (x[i] - min_val) / min_interval * min_interval + min_val
            target = y[i]
            bin_dict.setdefault(pos, [0, 0])

            if target == 1:
                bin_dict[pos][0] += 1
            else:
                bin_dict[pos][1] += 1

        print(bin_dict)
        #将字典转换为列表
        for key, val in bin_dict.items():
            t = [key]
            t.extend(val)
            bin_li.append(t)

        bin_li.sort(key=lambda x: x[0], reverse=False)
        #bin_li.sort(cmp=None, key=lambda x: x[0], reverse=False)
        print (bin_li)


        L_index = 0
        R_index = 1
        self.final_bin.append(bin_li[L_index][0])
        while True:
            L = bin_li[L_index]
            R = bin_li[R_index]


            # 计算分割点左边的信息熵
            p1 = L[1] / (L[1] + L[2] + 0.0)
            p0 = L[2] / (L[1] + L[2] + 0.0)

            if p1 <= 1e-5 or p0 <= 1e-5:
                LGain = 0
            else:
                LGain = -p1 * np.log(p1) - p0 * np.log(p0)

            #计算分割点右边的信息熵
            p1 = R[1] / (R[1] + R[2] + 0.0)
            p0 = R[2] / (R[1] + R[2] + 0.0)
            if p1 <= 1e-5 or p0 <= 1e-5:
                RGain = 0
            else:
                RGain = -p1 * np.log(p1) - p0 * np.log(p0)

            #计算整体信息熵
            p1 = (L[1] + R[1]) / (L[1] + L[2] + R[1] + R[2] + 0.0)
            p0 = (L[2] + R[2]) / (L[1] + L[2] + R[1] + R[2] + 0.0)

            if p1 <= 1e-5 or p0 <= 1e-5:
                ALLGain = 0
            else:
                ALLGain = -p1 * np.log(p1) - p0 * np.log(p0)


            #使用信息增益判断寻找分割点
            if np.absolute(ALLGain - LGain - RGain) <= self.min_epos: #最小信息增益作为分割标准
                # 如果未达到最小分割点，则合并区间
                bin_li[L_index][1] += R[1]
                bin_li[L_index][2] += R[2]
                R_index += 1

            else:
                #如果达到最小分割点，则继续判断下个分割点
                L_index = R_index
                R_index = L_index + 1
                self.final_bin.append(bin_li[L_index][0])

            #判断索引是否溢出
            if R_index >= len(bin_li):
                break

        print ('feature bin:', self.final_bin)


    def transform(self, x):
        res = []
        for e in x:
            index = self.get_Discretization_index(self.final_bin, e)
            res.append(index)

        res = np.asarray(res)
        return res


    '获取离散化索引'
    def get_Discretization_index(self, Discretization_vals, val):
        index = -1
        for i in range(len(Discretization_vals)):
            e = Discretization_vals[i]
            if val <= e:
                index = i
                break

        print(index)
        return index


obj=Feature_Discretization()
obj.fit(iris_df['col_1'],iris_df['target'],min_interval=4)


print(obj.transform(4))

#index=obj.transform(iris_df['col_1'])
#print(len(index))


