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


user_df=pd.read_excel('/Users/andpay/Documents/job/data/mode_data/refresh_data_merge.xlsx')
last_df=pd.read_excel('/Users/andpay/Documents/job/data/mode_data/initdata/datasource6_19.xlsx')
# marketing_df1=pd.read_excel('/Users/andpay/Documents/job/data/活动/activity_history/marketing_modedata3_14.xlsx')
# marketing_df2=pd.read_excel('/Users/andpay/Documents/job/data/活动/activity_history/marketing_modedata3_22.xlsx')
# /Users/andpay/Documents/job/data/mode_data/moderesult

#采用欠采样的方法进行样本的随机抽取
user_1=user_df[user_df['cate']==1]
user_0=user_df[user_df['cate']==0].sample(n=2*user_1.shape[0],random_state=1)
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

            split_point_cat[var ] = mid_dict

    return  dataframe


cate_list=['sex','brandcode','channel_type','marry','ccerate']


user_df=disper_split(user_df,cate_list)
last_df=disper_split(last_df,cate_list)
user_df=user_df.fillna(0)
last_df=last_df.fillna(0)

# marketing_df1=disper_split(marketing_df1,cate_list)
# marketing_df2=disper_split(marketing_df2,cate_list)
# marketing_df1=marketing_df1.fillna(0)
# marketing_df2=marketing_df2.fillna(0)

var_list=list(user_df.columns)
var_list.remove('partyid')
var_list.remove('name')
var_list.remove('cate')

#固定训练集和测试集数据
traindata, testdata= train_test_split(user_df,test_size=0.25,random_state=1)

x_train,y_train=traindata[var_list],traindata['cate']
x_test,y_test=testdata[var_list],testdata['cate']

print(var_list)

RFC = RandomForestClassifier(n_estimators=200,max_features=8,max_depth=6)
RFC.fit(x_train,y_train.astype(int))
rfc_col=list(np.round(RFC.feature_importances_,3))

GBC= GradientBoostingClassifier(n_estimators=200,learning_rate=0.03,max_depth=6,max_features=10)
GBC.fit(x_train,y_train.astype(int))
gbc_col=list(np.round(GBC.feature_importances_,3))

XGC=XGBClassifier(n_estimators=200,max_depth=8,learning_rate=0.03)
XGC.fit(x_train,y_train)
xgc_col=list(np.round(XGC.feature_importances_,3))


#feature_importanct=pd.DataFrame({'variable':var_list,'rfc':rfc_col,'gbc':gbc_col,'xgc':xgc_col})

#随机森林精确度验证
y_train_pred = RFC.predict(x_train)
y_test_pred = RFC.predict(x_test)

tree_train = accuracy_score(y_train, y_train_pred)
tree_test = accuracy_score(y_test, y_test_pred)
confs_rfc=metrics.confusion_matrix(y_test,y_test_pred)


print('Random Forest train/test accuracies %.3f/%.3f' % (tree_train, tree_test))
print('Random Forest train/test auc %.3f/%.3f' % (metrics.roc_auc_score(y_train,y_train_pred),
                                                  metrics.roc_auc_score(y_test,y_test_pred)))
print ('Random Forest train/test Recall:  %.3f/%.3f' % (metrics.recall_score(y_train,y_train_pred),
                                                   metrics.recall_score(y_test, y_test_pred)))
print ('Random Forest train/test precision:  %.3f/%.3f' % (metrics.precision_score(y_train,y_train_pred),
                                                   metrics.precision_score(y_test, y_test_pred)))


#结果交叉表
matric_rf_train=pd.crosstab(y_train,y_train_pred, rownames=['actual'], colnames=['preds'])
matric_rf_test=pd.crosstab(y_test,y_test_pred, rownames=['actual'], colnames=['preds'])
print(matric_rf_train)
print(matric_rf_test)


#GBDT精确度验证
y_train_pred_gbc = GBC.predict(x_train)
y_test_pred_gbc = GBC.predict(x_test)
tree_train = accuracy_score(y_train, y_train_pred_gbc)
tree_test = accuracy_score(y_test, y_test_pred_gbc)


print('Gradient Boosting train/test accuracies %.3f/%.3f' % (tree_train, tree_test))
print('Gradient Boosting train/test auc %.3f/%.3f' % (metrics.roc_auc_score(y_train,y_train_pred_gbc),
                                                      metrics.roc_auc_score(y_test, y_test_pred_gbc)))
print('Gradient Boosting train/test Recall %.3f/%.3f' % (metrics.recall_score(y_train,y_train_pred_gbc),
                                                      metrics.recall_score(y_test, y_test_pred_gbc)))
print('Gradient Boosting train/test precision %.3f/%.3f' % (metrics.precision_score(y_train,y_train_pred_gbc),
                                                      metrics.precision_score(y_test, y_test_pred_gbc)))


#结果交叉表
matric_gbc_train=pd.crosstab(y_train,y_train_pred_gbc, rownames=['actual'], colnames=['preds'])
matric_gbc_test=pd.crosstab(y_test,y_test_pred_gbc, rownames=['actual'], colnames=['preds'])
print(matric_gbc_train)
print(matric_gbc_test)


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


#把样本预测概率写进表中(proba返回的是概率数组，要使用切片提取相应类别的概率)
last_df['prob_rfc']=RFC.predict_proba(last_df[var_list])[:,1]
last_df['prob_gbc']=GBC.predict_proba(last_df[var_list])[:,1]
last_df['prob_xgc']=XGC.predict_proba(last_df[var_list])[:,1]

# marketing_df1['prob_rfc']=RFC.predict_proba(marketing_df1[var_list])[:,1]
# marketing_df1['prob_gbc']=GBC.predict_proba(marketing_df1[var_list])[:,1]
# marketing_df1['prob_xgc']=XGC.predict_proba(marketing_df1[var_list])[:,1]

# marketing_df2['prob_rfc']=RFC.predict_proba(marketing_df2[var_list])[:,1]
# marketing_df2['prob_gbc']=GBC.predict_proba(marketing_df2[var_list])[:,1]
# marketing_df2['prob_xgc']=XGC.predict_proba(marketing_df2[var_list])[:,1]
last_df=last_df.drop_duplicates(subset='partyid')


#print(last_df)
excel_writer=pd.ExcelWriter('/Users/andpay/Documents/job/data/mode_data/moderesult/moderesult6_19.xlsx',engine='xlsxwriter')
last_df.to_excel(excel_writer,'mode_result',index=False)
excel_writer.save()


# excel_writer=pd.ExcelWriter('/Users/andpay/Desktop/marketing_rfc_gbc_xgb.xlsx',engine='xlsxwriter')
# last_df.to_excel(excel_writer,'mode_result4_10',index=False)
# marketing_df1.to_excel(excel_writer,'mode_result3_14',index=False)
# marketing_df2.to_excel(excel_writer,'mode_result3_22',index=False)
# excel_writer.save()
