
'''
基于隔离森林的欺诈检测
'''

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score,confusion_matrix,f1_score
# from function_package.data_convert_function import data_convert_onehot,data_convert_from_dict


'''对数据进行多次训练'''
def train(df_data,clf,ensembleSize=5,sampleSize=10000):
    mdlLst=[]
    for n in range(ensembleSize):
        X=df_data.sample(sampleSize)
        clf.fit(X)
        mdlLst.append(clf)

    return mdlLst



'''统一的预测函数'''
def predict(X,mdlLst):
    y_pred=np.zeros(X.shape[0])
    for clf in mdlLst:
        #返回异常分
        y_pred=np.add(y_pred,clf.decision_function(X).reshape(X.shape[0],))

    y_pred=(y_pred*1.0)/len(mdlLst)

    return y_pred


'''对数据进行独热编码'''
def data_convert_onehot(dataframe,cate_list,return_status='df'):
    dummy_map = {}
    dummy_columns = []
    for raw_col in cate_list:
        dummies = pd.get_dummies(dataframe.loc[:, raw_col], prefix=raw_col)
        col_onehot = pd.concat([dataframe[raw_col], dummies], axis=1)
        col_onehot = col_onehot.drop_duplicates()
        dataframe = pd.concat([dataframe, dummies], axis=1)
        del dataframe[raw_col]
        dummy_map[raw_col] = col_onehot
        dummy_columns = dummy_columns + list(dummies)

    if return_status=='df':
        return dataframe,dummy_map
    elif return_status=='hot_data':
        return dummy_map
    else:
        raise ValueError("return_status input error")



"""
'''基于行为序列的隔离森林欺诈检测'''

#行为序列数据
data_verify=pd.read_excel('/Users/admin/Documents/data_analysis/fraud_model/analysis_behave/user_visit_behave_transform_190103.xlsx')
test_dataset=pd.read_excel('/Users/admin/Documents/data_analysis/fraud_model/analysis_behave/user_visit_behave_79_transform.xlsx')


#欺诈样本
fraud_dataset=pd.read_excel('/Users/admin/Documents/data_analysis/fraud_model/data_source/address_data/fraud_user_data.xlsx')
test_dataset.loc[test_dataset['id_user'].isin(fraud_dataset['id_user']),'fraud_cate']=1
test_dataset['fraud_cate']=test_dataset['fraud_cate'].fillna(0)
print(test_dataset.shape[0])


#剔除非欺诈M2数据
del_no=test_dataset[(test_dataset['user_cate']=='M2')&(test_dataset['fraud_cate']!=1)]['trade_no']
test_dataset=test_dataset[~test_dataset['trade_no'].isin(del_no)]
print(test_dataset.shape[0])


# 标签转换
test_dataset.loc[test_dataset['user_cate']!='M2','user_cate']=0
test_dataset.loc[test_dataset['user_cate']=='M2','user_cate']=1
print(test_dataset['user_cate'].value_counts())

data_verify.loc[data_verify['user_cate']!='M2','user_cate']=0
data_verify.loc[data_verify['user_cate']=='M2','user_cate']=1

verify_y=data_verify['user_cate']
verify_x=data_verify.drop(['trade_no','id_user','user_cate'],1)

y_true=test_dataset['user_cate']
df_data=test_dataset.drop(['trade_no','id_user','user_cate','fraud_cate'],1)
X_train, X_test, y_train, y_test = train_test_split(df_data, y_true, test_size=0.3, random_state=42)


'''基于隔离森林的异常检测'''
alg=IsolationForest(n_estimators=100, max_samples='auto', contamination=0.01,
                    max_features=1.0, bootstrap=False, n_jobs=-1, random_state=42, verbose=0,behaviour="new")

if_mdlLst=train(df_data,alg)
# print(if_mdlLst)

if_y_pred=predict(df_data,if_mdlLst)
if_y_pred=1-if_y_pred
print(if_y_pred)


# 验证数据预测
verify_pred=predict(verify_x,if_mdlLst)
verify_pred=1-verify_pred


#Creating class labels based on decision function
# 把预测结果排名前95%部分设为1 其余设为0
if_y_pred_class=if_y_pred.copy()
print(np.percentile(if_y_pred_class,95))
if_y_pred_class[if_y_pred>=np.percentile(if_y_pred,95)]=1
if_y_pred_class[if_y_pred<np.percentile(if_y_pred,95)]=0


# 验证数据预测结果分类
verify_pred_class=verify_pred.copy()
print(np.percentile(verify_pred_class,95))
verify_pred_class[verify_pred>=0.895]=1
verify_pred_class[verify_pred<0.895]=0


#训练数据结果
print(roc_auc_score(y_true, if_y_pred_class))
print(f1_score(y_true, if_y_pred_class))
if_cm=confusion_matrix(y_true, if_y_pred_class)
print(if_cm[0][0],if_cm[0][1])
print(if_cm[1][0],if_cm[1][1])

print(roc_auc_score(verify_y,verify_pred_class))
print(f1_score(verify_y,verify_pred_class))
if_cm_verify=confusion_matrix(verify_y,verify_pred_class)
print(if_cm_verify[0][0],if_cm_verify[0][1])
print(if_cm_verify[1][0],if_cm_verify[1][1])

"""


'''基于隔离森林的整体数据欺诈检测'''
dataset=pd.read_excel('/Users/admin/Documents/data_analysis/fraud_model/fraud_analysis/fraud_user_character_merge.xlsx')
# test_dataset=pd.read_excel('/Users/admin/Documents/data_analysis/fraud_model/fraud_analysis/fraud_user_character_test.xlsx')
test_dataset=pd.read_excel('/Users/admin/Documents/data_analysis/fraud_model/fraud_analysis/fraud_user_character_test1934.xlsx')
col_list=['age','sex','zm_face','zm_watchlist','apply_num_1day','user_cancel_num_1day']
# print(dataset.columns)


dataset['sex']=dataset['sex'].fillna(2)
dataset['capacity']=dataset['capacity'].apply(lambda x: str(x).split('+')[1] if str(x).find('+') !=-1
                                         else str(x).split('+')[0] )
dataset['capacity']=dataset['capacity'].apply(lambda x:str(x).replace('G',''))
test_dataset['sex']=test_dataset['sex'].fillna(2)
dataset=dataset.fillna(0)

test_dataset['capacity']=test_dataset['capacity'].apply(lambda x: str(x).split('+')[1] if str(x).find('+') !=-1
                                         else str(x).split('+')[0] )
test_dataset['capacity']=test_dataset['capacity'].apply(lambda x:str(x).replace('G',''))
test_dataset=test_dataset.fillna(0)
print(dataset['target'].value_counts())


# 标签转换
test_dataset.loc[test_dataset['user_cate']!='M2','user_cate']=0
test_dataset.loc[test_dataset['user_cate']=='M2','user_cate']=1


train_y=dataset['target']
train_data=dataset[col_list]
test_y=test_dataset['user_cate']
test_data=test_dataset[col_list]
print('node 1 over')

train_data,train_columns=data_convert_onehot(train_data,cate_list=['zm_face','zm_watchlist','sex'])
test_data,test_columns=data_convert_onehot(test_data,cate_list=['zm_face','zm_watchlist','sex'])
# print(train_data.columns)

'''基于隔离森林的异常检测'''
alg=IsolationForest(n_estimators=200, max_samples='auto', contamination=0.01,
                    max_features=1.0, bootstrap=False, n_jobs=-1, random_state=42, verbose=0,behaviour="new")

alg.fit(train_data)

# if_mdlLst=train(train_data,alg)
# print(if_mdlLst)
# if_y_pred=predict(train_data,if_mdlLst)

if_y_pred=alg.decision_function(train_data)
if_y_pred=1-if_y_pred
print(if_y_pred)

# 验证数据预测
# verify_pred=predict(test_data,if_mdlLst)
verify_pred=alg.decision_function(test_data)
verify_pred=1-verify_pred


#Creating class labels based on decision function
# 把预测结果排名前95%部分设为1 其余设为0
if_y_pred_class=if_y_pred.copy()
print(np.percentile(if_y_pred,95))  #95%分位数值
if_y_pred_class[if_y_pred>=np.percentile(if_y_pred,95)]=1
if_y_pred_class[if_y_pred<np.percentile(if_y_pred,95)]=0

# 验证数据预测结果分类
verify_pred_class=verify_pred.copy()
print(np.percentile(verify_pred_class,95))
verify_pred_class[verify_pred>=np.percentile(if_y_pred,95)]=1
verify_pred_class[verify_pred<np.percentile(if_y_pred,95)]=0

#训练数据结果
print(roc_auc_score(train_y, if_y_pred_class))
print(f1_score(train_y, if_y_pred_class))
if_cm=confusion_matrix(train_y, if_y_pred_class)
print(if_cm[0][0],if_cm[0][1])
print(if_cm[1][0],if_cm[1][1])

print(roc_auc_score(test_y,verify_pred_class))
print(f1_score(test_y,verify_pred_class))
if_cm_verify=confusion_matrix(test_y,verify_pred_class)
print(if_cm_verify[0][0],if_cm_verify[0][1])
print(if_cm_verify[1][0],if_cm_verify[1][1])





