#!/usr/bin/python
# encoding=utf-8

import pandas as pd
import numpy as np
from scipy import stats
import requests
from sklearn.datasets import load_iris
import json

'''
get 模式下调用
'''
# iris=load_iris()
#
# for row in iris.data:
#     row_s=''
#     for i in range(len(row)):
#         if i==0 or i==len(row):
#             row_s=row_s+str(row[i])
#         else:
#             row_s=row_s+','+str(row[i])
#
#     url = 'http://127.0.0.1:5003/?inputdata='
#     base=url+row_s
#     print(base)
#     response = requests.get(base)
#     answer = response.json()
#     print('预测结果',answer)



'''
post 模式下调用
'''

urls='http://127.0.0.1:3000/web'
user_info = {'name': 'letian', 'password': '123'}
response = requests.post(url=urls, data='ss')

print(response.text)



