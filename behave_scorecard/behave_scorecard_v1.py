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


dataset=pd.read_excel('/Users/andpay/Documents/job/model/behave_model/behave_model_dataset_v1.xlsx')
var_list=list(dataset.columns)
model_var_list=remove_list(var_list,['partyid','loanid','last_overday','cate'])


category_var=['sex','city_id','channel_type','brandcode']
continue_var=remove_list(model_var_list,category_var)  #这里会改变newvar_list的元素数量
newuser_dataset=disper_split(dataset,category_var)
newuser_dataset[continue_var]=newuser_dataset[continue_var].fillna(0)

#目标样本索引
#index_zero=dataset[dataset['cate']==0].index
index_one=dataset[dataset['cate']==1].index

#类别型变量进行独热编码
dataset_category=dataset[category_var]
encoder=preprocessing.OneHotEncoder()
encoder.fit(dataset_category)
dataset_encoder=encoder.transform(dataset_category).toarray()


#自变量标准化处理，方便进行smote过采样
dataset_continue=preprocessing.scale(newuser_dataset[continue_var],axis=1)


#合并结果
dataset_merge=np.hstack([dataset_encoder,dataset_continue])
# dataset_merge_index=dataset_merge[index_one,:]
# print(dataset_merge_index.shape)


#对少数类样本进行smote过采样
over_sample=SMOTE()
x,y=over_sample.fit_sample(dataset_merge,dataset['cate'])
print(x.shape)
print(y.shape)


#固定训练集和测试集数据
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.25,random_state=1)

#超参数随机搜索
# parameter_tree = {'max_depth': range(3, 10), 'n_estimators': range(100, 200, 10), 'learning_rate': [0.01, 0.02, 0.03]}
# rds=RandomizedSearchCV(XGBClassifier(),parameter_tree,cv=KFold(n_splits=5),scoring='recall',n_iter=10)
# rds.fit(x_train,y_train)
# print ("best score: {}".format(rds.best_score_))
# print(rds.best_params_)



XGC=XGBClassifier(n_estimators=150,max_depth=9,learning_rate=0.03)
XGC.fit(x_train,y_train)
xgc_col=list(np.round(XGC.feature_importances_,3))


#变量重要性排序
# var_importance=pd.DataFrame({'var':model_var_list,'importance_value':xgc_col})
# var_importance=var_importance.sort_values(by='importance_value',ascending=0)
# print(var_importance)


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



end_time=time.time()

print('\t')
print('程序运行时间：%s 秒'%(round(end_time-start_time)))