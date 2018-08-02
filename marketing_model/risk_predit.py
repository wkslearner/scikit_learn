#!/usr/bin/python
# encoding=utf-8

import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn import metrics
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import RandomOverSampler,SMOTE,ADASYN

'''分类变量处理'''
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


'''移除列表中部分元素，生成新列表'''
def remove_list(all_list,remove_element):
    for i in remove_element:
        all_list.remove(i)

    return all_list


#读取训练样本集
risk_dataset=pd.read_excel('/Users/andpay/Documents/job/mode/marketing_model/risk_dataset.xlsx')
var_list=list(risk_dataset.columns)
var_list.remove('partyid')
var_list.remove('last_overday')


#目标变量处理
risk_dataset.loc[risk_dataset['last_overday']>=30,'target_var']=1
risk_dataset.loc[risk_dataset['last_overday']<30,'target_var']=0


#自变量处理
category_var=['sex','city_id','brandcode','channel_type']
continue_var=remove_list(var_list,category_var)
risk_dataset=disper_split(risk_dataset,category_var)
risk_dataset[continue_var]=risk_dataset[continue_var].fillna(-1)


# 逾期分布
target_distribute_first=risk_dataset['partyid'].groupby(risk_dataset['target_var']).count()
print(target_distribute_first)


#采用欠采样的方法进行样本的随机抽取
# user_1=risk_dataset[risk_dataset['target_var']==1]
# user_0=risk_dataset[risk_dataset['target_var']==0].sample(n=2*user_1.shape[0],random_state=1)
# risk_dataset=pd.concat([user_0,user_1],axis=0)


#smote过采样
x=risk_dataset[var_list]
y=risk_dataset['target_var']
sm = SMOTE()
x_resampled, y_resampled = sm.fit_sample(x, y)
new_dataset=pd.DataFrame(x_resampled,columns=var_list)
new_dataset['target_var']=y_resampled

# 逾期分布
target_distribute=new_dataset['target_var'].groupby(new_dataset['target_var']).count()
print(target_distribute)

#固定训练集和测试集数据
traindata, testdata= train_test_split(new_dataset,test_size=0.25,random_state=1)
x_train,y_train=traindata[var_list],traindata['target_var']
x_test,y_test=testdata[var_list],testdata['target_var']


XGC=XGBClassifier(n_estimators=200,max_depth=8,learning_rate=0.03)
XGC.fit(x_train,y_train)
xgc_col=list(np.round(XGC.feature_importances_,3))


var_importance=pd.DataFrame({'var':var_list,'importance_value':xgc_col})
var_importance=var_importance.sort_values(by='importance_value',ascending=0)
print(var_importance)


#XGboost精确度验证
y_train_pred_xgb = XGC.predict(x_train)
y_test_pred_xgb = XGC.predict(x_test)
tree_train = accuracy_score(y_train, y_train_pred_xgb)
tree_test = accuracy_score(y_test, y_test_pred_xgb)


print('XG Boosting train/test accuracies %.3f/%.3f' % (tree_train, tree_test))
print('XG Boosting train/test auc %.3f/%.3f' % (metrics.roc_auc_score(y_train,y_train_pred_xgb),
                                                metrics.roc_auc_score(y_test, y_test_pred_xgb)))
print('XG Boosting train/test Recall %.3f/%.3f' % (metrics.recall_score(y_train,y_train_pred_xgb),
                                                   metrics.recall_score(y_test, y_test_pred_xgb)))
print('XG Boosting train/test precision %.3f/%.3f' % (metrics.precision_score(y_train,y_train_pred_xgb),
                                                      metrics.precision_score(y_test, y_test_pred_xgb)))



#结果交叉表
matric_xgb_train=pd.crosstab(y_train,y_train_pred_xgb, rownames=['actual'], colnames=['preds'])
matric_xgb_test=pd.crosstab(y_test,y_test_pred_xgb, rownames=['actual'], colnames=['preds'])
print(matric_xgb_train)
print(matric_xgb_test)




