#!/usr/bin/python
# encoding=utf-8

from sklearn import preprocessing
import numpy as np


x = np.array([
    [11., -31., 25.],
    [20., 60., 30.],
    [-5., 15., -18.]])


'''指定均值方差数据标准化(默认均值0 方差 1)'''
x_scaled = preprocessing.scale(x,axis=1)
print(x_scaled)

mean_row=x_scaled.mean(axis=1)
std_row= x_scaled.std(axis=1)

print(mean_row)
print(std_row)

x_scaled = preprocessing.scale(x,axis=1)
mean_column=x_scaled.mean(axis=1)
std_column=x_scaled.std(axis=1)


'''StandardScaler类,可以保存训练集中的参数'''
scaler = preprocessing.StandardScaler().fit(x)

'''标准化前均值和方差'''
scaler_mean=scaler.mean_ ;scaler_std=scaler.scale_

'''标准化后的矩阵'''
scaler_matrix=scaler.transform(x)


'''数据归一化，把数据映射到最大值和最小值之间'''
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,10))
min_max_matrix=min_max_scaler.fit_transform(x)