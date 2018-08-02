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


data_df=pd.read_excel('/Users/andpay/Documents/job/mode/leave_mode/data_source.xlsx')
data_df=data_df.fillna(0)

data_df[data_df['sex']=='男']=1
data_df[data_df['sex']=='女']=0

var_list=list(data_df.columns)
var_list.remove('partyid')
var_list.remove('user_cate')

#固定训练集和测试集数据
traindata, testdata= train_test_split(data_df,test_size=0.25,random_state=1)

x_train,y_train=traindata[var_list],traindata['user_cate']
x_test,y_test=testdata[var_list],testdata['user_cate']

RFC = RandomForestClassifier(n_estimators=200,max_features=8,max_depth=6)
RFC.fit(x_train,y_train.astype(int))
rfc_col=list(np.round(RFC.feature_importances_,3))
var_inportance_df=pd.DataFrame({'variable':var_list,'inportance_rank':rfc_col})
var_inportance_df=var_inportance_df.sort_values(by='inportance_rank',ascending=0)
print(var_inportance_df)


#随机森林精确度验证
y_train_pred = RFC.predict(x_train)
y_test_pred = RFC.predict(x_test)

tree_train = accuracy_score(y_train, y_train_pred)
tree_test = accuracy_score(y_test, y_test_pred)
confs_rfc=metrics.confusion_matrix(y_test,y_test_pred)

print(confs_rfc)


print('Random Forest train/test accuracies %.3f/%.3f' % (tree_train, tree_test))

print('Random Forest train/test auc %.3f/%.3f' % (metrics.roc_auc_score(y_train,y_train_pred),
                                                  metrics.roc_auc_score(y_test,y_test_pred)))

print ('Random Forest train/test Recall:  %.3f/%.3f' % (metrics.recall_score(y_train,y_train_pred),
                                                   metrics.recall_score(y_test, y_test_pred)))

print ('Random Forest train/test precision:  %.3f/%.3f' % (metrics.precision_score(y_train,y_train_pred),
                                                   metrics.precision_score(y_test, y_test_pred)))

