#!/usr/bin/python
# encoding=utf-8

import pandas as pd
from data_process.list_process import remove_list
from data_process.feature_handle import disper_split
from sklearn import preprocessing


dataset=pd.read_excel('/Users/andpay/Documents/job/model/behave_model/behave_model_dataset_v1.xlsx')
var_list=list(dataset.columns)
model_var_list=remove_list(var_list,['partyid','loanid','last_overday','cate'])


category_var=['sex','city_id','channel_type','brandcode']
continue_var=remove_list(model_var_list,category_var)  #这里会改变newvar_list的元素数量
newuser_dataset=disper_split(dataset,category_var)
newuser_dataset[continue_var]=newuser_dataset[continue_var].fillna(0)
print(model_var_list)
print(dataset.shape)


#类别型变量进行独热编码
dataset_category=dataset[category_var]
encoder=preprocessing.OneHotEncoder()
encoder.fit(dataset_category)
dataset_encoder=encoder.transform(dataset_category).toarray()
print(dataset_encoder.shape)


#自变量标准化处理，方便进行smote过采样
dataset_continue=preprocessing.scale(newuser_dataset[continue_var])

dataset_merge=pd.concat([dataset_category,dataset_continue],axis=1)

