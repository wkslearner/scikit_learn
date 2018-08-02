#!/usr/bin/python
# encoding=utf-8

import pandas as pd
import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from statsmodels.stats.outliers_influence import variance_inflation_factor
from score_card_version1.woe_information import *
import statsmodels.api as sm
from score_card_version1.result_check import KS_AR
from sklearn.metrics import roc_curve,auc,roc_auc_score
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import metrics
from xgboost.sklearn import XGBClassifier
import xgboost as xgb


user_df=pd.read_excel('/Users/andpay/Documents/job/data/活动/4.18帮还名单/test_file.xlsx')
user_df=user_df[0:10]


'''非数字分类变量处理'''
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


cate_list=['sex','brandcode','channel_type','marry','ccerate']

user_df=disper_split(user_df,cate_list)
user_df=user_df.fillna(0)
var_list=list(user_df.columns)
var_list.remove('partyid')
var_list.remove('name')
#var_list.remove('cate')

XGC=XGBClassifier(n_estimators=200,max_depth=8,learning_rate=0.03)
user_df['prob_xgc']=XGC.predict_proba(user_df[var_list])[:,1]

print(user_df)