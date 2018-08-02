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
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import GridSearchCV,cross_val_score
import matplotlib.pylab as plt


# user_df=pd.read_excel('/Users/andpay/Documents/job/data/mode_data/modedata_3_28.xlsx','data')
# partyid_df=pd.read_excel('/Users/andpay/Documents/job/data/mode_data/modedata_3_28.xlsx','id')
# partyid_df['cate']=1
# user_df=pd.merge(user_df,partyid_df,on='partyid',how='left')
# last_df=pd.read_excel('/Users/andpay/Documents/job/data/mode_data/filter_data3_30.xlsx')


id_list=['1012222000766459','1012222000776201','1012222000788446','1012222000806482','1012222000395680',
         '1012222000439546','1012222000806927','1012222000461962','1012222000822555','1012222000892648',
         '1012222000885859','1012222000785173','1012222000893996','1012222000880991','1012222000869762',
         '1012222000612831','1012222000429243','1012222000789130','1012222000006747','1012222000389967',
         '1012222000863300','1012222000627093','1012222000811536','1012222000757780','1012222000542331']

user_df=pd.read_excel('/Users/andpay/Documents/job/data/mode_data/refresh_data_merge.xlsx')


#采用欠采样的方法进行样本的随机抽取
user_1=user_df[user_df['cate']==1]
user_0=user_df[user_df['cate']==0].sample(n=2*user_1.shape[0],random_state=1)
print(user_1.shape[0])
user_df=pd.concat([user_0,user_1],axis=0)


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

            split_point_cat[var] = mid_dict

    return  dataframe


cate_list=['sex','brandcode','channel_type','marry','ccerate']

user_df=disper_split(user_df,cate_list)
#last_df=disper_split(last_df,cate_list)

# for key in cate_list:
#     user_df[key]=user_df[key].fillna('null')

user_df=user_df.fillna(0)
#last_df=last_df.fillna(0)

var_list=list(user_df.columns)
var_list.remove('partyid')
var_list.remove('name')

end_list=var_list
end_list.remove('cate')

traindata, testdata= train_test_split(user_df,test_size=0.25,random_state=1)
x_train,y_train=traindata[var_list],traindata['cate']
x_test,y_test=testdata[var_list],testdata['cate']
#print(user_df)

dtrain=xgb.DMatrix(x_train,label=y_train)
dtest=xgb.DMatrix(x_test)


params={'booster':'gbtree',  #基分类器可选参数gbtree和gblinear
    'eval_metric': 'auc',  #模型评价指标
    'min_child_weight':1,  #
    'subsample':0.7,     #训练样本占总样本的比例，防止过拟合
    'max_depth':7,
    'colsample_bytree':0.7,    #特征的随机采样比例
    'eta': 0.03,    #学习步长，
    'nthread':8,    #最大线程数，默认为模型所能获取到最大的线程数
    'silent':1}     #是否打印模型信息，0表示打印，1表示不打印

watchlist = [(dtrain,'train')]

#训练过程
bst=xgb.train(params,dtrain,num_boost_round=200,evals=watchlist)
ypred=bst.predict(dtest)

#last_df['prob']=bst.predict(xgb.DMatrix(last_df[end_list]))


# 设置阈值,输出一些评价指标
y_pred = (ypred >= 0.5)*1


'''模型评价'''
print('测试结果数据')
#auc指标——为roc曲线下方的面积
print ('AUC: %.4f' % metrics.roc_auc_score(y_test,ypred))
#准确率
print ('ACC: %.4f' % metrics.accuracy_score(y_test,y_pred))
#召回率
print ('Recall: %.4f' % metrics.recall_score(y_test,y_pred))
#F1为准确率和召回率的中和指标
print ('F1-score: %.4f' %metrics.f1_score(y_test,y_pred))
#精确度
print ('Precesion: %.4f' %metrics.precision_score(y_test,y_pred))


#混淆矩阵
confs=pd.crosstab(y_test,y_pred, rownames=['actual'], colnames=['preds'])
#confs=metrics.confusion_matrix(y_test,y_pred,labels=[0,1])
print(confs)


# excel_writer=pd.ExcelWriter('/Users/andpay/Desktop/portrait_xgboost.xlsx',engine='xlsxwriter')
# last_df.to_excel(excel_writer,'mode_result',index=False)
# excel_writer.save()
