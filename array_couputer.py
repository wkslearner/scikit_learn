#!/usr/bin/python
# encoding=utf-8

import pandas as pd
import numpy as np
from data_process.list_process import remove_list
from data_process.feature_handle import disper_split
from sklearn import preprocessing
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV, KFold ,RandomizedSearchCV
from sklearn.cross_validation import train_test_split
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import make_scorer, fbeta_score, accuracy_score,recall_score
from sklearn import metrics
import time


start_time=time.time()

dataset=pd.read_excel('/Users/andpay/Documents/job/model/behave_model/behave_model_dataset_v1_1.xlsx')
dataset.loc[dataset['last_overday']>=10,'cate']=1
dataset.loc[dataset['last_overday']<10,'cate']=0


var_list=list(dataset.columns)
model_var_list=remove_list(var_list,['partyid','loanid','last_overday','cate','register_duration'])


category_var=['sex','city_id','channel_type','brandcode']
continue_var=remove_list(model_var_list,category_var)  #这里会改变newvar_list的元素数量
newuser_dataset=disper_split(dataset,category_var)
newuser_dataset[continue_var]=newuser_dataset[continue_var].fillna(0)


# x_train,x_test,y_train,y_test= train_test_split(,test_size=0.25,random_state=1)
XGC=XGBClassifier(n_estimators=150,max_depth=9,learning_rate=0.03)
XGC.fit(newuser_dataset[continue_var].astype(int),dataset['cate'].astype(int))
xgc_col=list(np.round(XGC.feature_importances_,3))


#变量重要性排序
var_importance=pd.DataFrame({'var':continue_var,'importance_value':xgc_col})
var_importance=var_importance.sort_values(by='importance_value',ascending=0)
print(var_importance)







