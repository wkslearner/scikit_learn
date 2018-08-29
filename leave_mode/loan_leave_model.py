#!/usr/bin/python
# encoding=utf-8

import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import make_scorer,recall_score,accuracy_score,roc_auc_score
from sklearn.model_selection import RandomizedSearchCV,KFold
from sklearn import metrics
import time
from  data_sample.SMOTE import Smote


start_time=time.time()


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
    end_list=[]
    for element in all_list:
        if element in remove_element:
            continue
        else:
            end_list.append(element)

    return end_list



'''空值检测，如果数据中空值超过一半或变量只有一个取值时，返回数据不可用'''
def check_nullvalue(dataframe):

    '''
    :param dataframe: 目标数据框
    :return: 不能用的变量列表
    '''
    column=list(dataframe.columns)

    use_list = []
    unuse_list = []
    for key in column:
        if dataframe[dataframe[key].isnull()].shape[0]<len(dataframe[key])/2:
            use_list.append(key)
        elif len(dataframe[dataframe[key].notnull()][key].unique())<2:
            unuse_list.append(key)
        else:
            unuse_list.append(key)

    return unuse_list


#随机欠采样
def random_under_sample(df,sample_col,majority_label,minority_label,sample_coefficient,rd_state=1):
    '''
    :param df: 目标数据框
    :param sample_col: 多数类和少数类区分列
    :param majority_label: 多数类标签
    :param minority_label: 少数类标签
    :param sample_coefficient: 采样后，多数类和少数类的比例
    :param rd_state: 随机种子
    :return: 
    '''

    user_1 = df[df[sample_col] == minority_label]
    user_0 = df[df[sample_col] == majority_label].sample(n= sample_coefficient* user_1.shape[0], random_state=rd_state)
    result_df = pd.concat([user_0, user_1], axis=0)

    return result_df



loan_dataset=pd.read_excel('/Users/andpay/Documents/job/model/loan_leave_model/loan_leave_dataset_v2.xlsx')
loan_dataset=random_under_sample(df=loan_dataset,sample_col='cate_renew',majority_label=0,minority_label=1,
                                 sample_coefficient=2)

first_target_dataset=pd.read_excel('/Users/andpay/Documents/job/model/loan_leave_model/model_practice/model_practice_v2.xlsx')
first_target_dataset_predit=first_target_dataset.drop(['renew_yesornot'],axis=1)
var_list=list(loan_dataset.columns)
var_list=remove_list(var_list,['partyid','loanid','diff_time','cate','cate_renew'])
print(var_list)

target_dataset=first_target_dataset_predit.copy()


#自变量处理
category_var=['sex','city_id','brandcode','channel']
continue_var=remove_list(var_list,category_var)  #这里会改变newvar_list的元素数量
newuser_dataset=disper_split(loan_dataset,category_var)
newuser_dataset[continue_var]=newuser_dataset[continue_var].fillna(-1)


target_dataset=disper_split(target_dataset,category_var)
target_dataset[continue_var]=target_dataset[continue_var].fillna(-1)


newuser_dataset=newuser_dataset[var_list+['cate_renew']].apply(pd.to_numeric)
target_dataset=target_dataset[var_list].apply(pd.to_numeric)


#固定训练集和测试集数据
traindata, testdata= train_test_split(newuser_dataset,test_size=0.25,random_state=1)
x_train,y_train=traindata[var_list],traindata['cate_renew']
x_test,y_test=testdata[var_list],testdata['cate_renew']


#超参数随机搜索
# parameter_tree = {'max_depth': range(3, 10),'n_estimators':range(100,200,10),'learning_rate':[0.01,0.02,0.03]}
# rds=RandomizedSearchCV(XGBClassifier(),parameter_tree,cv=KFold(n_splits=5),scoring='recall',n_iter=10)
# rds.fit(x_train,y_train)
# print ("best score: {}".format(rds.best_score_))
# print(rds.best_params_)


XGC=XGBClassifier(n_estimators=160,max_depth=7,learning_rate=0.03,reg_alpha=30)
XGC.fit(x_train,y_train)
xgc_col=list(np.round(XGC.feature_importances_,3))


#变量重要性排序
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


newuser_dataset['prob_xgc']=XGC.predict_proba(newuser_dataset[var_list])[:,1]
first_target_dataset['prob_xgc']=XGC.predict_proba(target_dataset[var_list])[:,1]
#first_target_dataset=first_target_dataset.astype(str)

# excel_writer=pd.ExcelWriter('/Users/andpay/Documents/job/model/loan_leave_model/model_practice/model_practice_v2_predict1.xlsx',engine='xlsxwriter')
# first_target_dataset.to_excel(excel_writer,'mode_result',index=False)
# excel_writer.save()


end_time=time.time()
print('\t')
print('程序运行时间：%s 秒'%(round(end_time-start_time)))





