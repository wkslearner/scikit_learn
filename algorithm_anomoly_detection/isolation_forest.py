
'''
基于隔离森林的异常检测算法
'''

import pandas as pd
import numpy as np
import math
import random
import random
from matplotlib import pyplot
import os
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor ## Only available with scikit-learn 0.19 and later
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix,f1_score
import seaborn as sn


# print(os.listdir("../input"))
df=pd.read_csv('creditcard.csv')

#查看欺诈样本数量
print(df[df['Class']==1].shape[0])

y_true=df['Class']
df_data=df.drop('Class',1)
X_train, X_test, y_train, y_test = train_test_split(df_data, y_true, test_size=0.3, random_state=42)


def preprocess(df_data):
    for col in df_data:
        df_data[col]=(df_data[col]-np.min(df_data[col]))/(np.max(df_data[col])-np.min(df_data[col]))

    return df_data


'''对数据进行多次训练'''
def train(X,clf,ensembleSize=5,sampleSize=10000):
    mdlLst=[]
    for n in range(ensembleSize):
        X=df_data.sample(sampleSize)
        clf.fit(X)
        mdlLst.append(clf)

    return mdlLst


'''统一的预测函数,'''
def predict(X,mdlLst):
    y_pred=np.zeros(X.shape[0])
    for clf in mdlLst:
        y_pred=np.add(y_pred,clf.decision_function(X).reshape(X.shape[0],))
    y_pred=(y_pred*1.0)/len(mdlLst)

    return y_pred


'''基于隔离森林的异常检测'''
alg=IsolationForest(n_estimators=100, max_samples='auto', contamination=0.01,
                    max_features=1.0, bootstrap=False, n_jobs=-1, random_state=42, verbose=0,behaviour="new")

if_mdlLst=train(X_train,alg)
print(if_mdlLst)

if_y_pred=predict(X_test,if_mdlLst)
if_y_pred=1-if_y_pred
print(if_y_pred)


#Creating class labels based on decision function
# 把预测结果排名前95%部分设为1 其余设为0
if_y_pred_class=if_y_pred.copy()
if_y_pred_class[if_y_pred>=np.percentile(if_y_pred,95)]=1
if_y_pred_class[if_y_pred<np.percentile(if_y_pred,95)]=0

# 主要指标信息
roc_auc_score(y_test, if_y_pred_class)
f1_score(y_test, if_y_pred_class)
if_cm=confusion_matrix(y_test, if_y_pred_class)


df_cm = pd.DataFrame(if_cm,['True Normal', 'True Fraud'], ['Pred Normal', 'Pred Fraud'])
pyplot.figure(figsize=(8, 4))
sn.set(font_scale=1.4)   #for label  size
sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, fmt='g')  #font size

'''基于k-means的异常检测'''
kmeans = KMeans(n_clusters=8,random_state=42,n_jobs=-1).fit(X_train)

X_test_clusters=kmeans.predict(X_test)
X_test_clusters_centers=kmeans.cluster_centers_
dist = [np.linalg.norm(x-y) for x,y in zip(X_test.values,X_test_clusters_centers[X_test_clusters])]

km_y_pred=np.array(dist)
km_y_pred[dist>=np.percentile(dist,95)]=1
km_y_pred[dist<np.percentile(dist,95)]=0

X_test_clusters=kmeans.predict(X_test)
X_test_clusters_centers=kmeans.cluster_centers_
dist = [np.linalg.norm(x-y) for x,y in zip(X_test.values,X_test_clusters_centers[X_test_clusters])]


km_y_pred=np.array(dist)
km_y_pred[dist>=np.percentile(dist,95)]=1
km_y_pred[dist<np.percentile(dist,95)]=0

# 指标计算
roc_auc_score(y_test, km_y_pred)
f1_score(y_test, km_y_pred)
km_cm=confusion_matrix(y_test, km_y_pred)


df_cm = pd.DataFrame(km_cm,['True Normal','True Fraud'],['Pred Normal','Pred Fraud'])
pyplot.figure(figsize = (8,4))
sn.set(font_scale=1.4)#for label size
sn.heatmap(df_cm, annot=True,annot_kws={"size": 16},fmt='g')# font size


'''基于异常因子的异常检测'''
clf=LocalOutlierFactor(n_neighbors=20, algorithm='auto', leaf_size=30,
                   metric='minkowski', p=2, metric_params=None, contamination=0.1, n_jobs=-1)

clf.fit(X_test)
lof_y_pred=clf.negative_outlier_factor_

#Creating class labels based on decision function
lof_y_pred_class=lof_y_pred.copy()
lof_y_pred_class[lof_y_pred>=np.percentile(lof_y_pred,95)]=1
lof_y_pred_class[lof_y_pred<np.percentile(lof_y_pred,95)]=0

roc_auc_score(y_test, lof_y_pred_class)
f1_score(y_test, lof_y_pred_class)
lof_cm=confusion_matrix(y_test, lof_y_pred_class)

df_cm = pd.DataFrame(lof_cm,['True Normal','True Fraud'],['Pred Normal','Pred Fraud'])
pyplot.figure(figsize = (8,4))
sn.set(font_scale=1.4) #for label size
sn.heatmap(df_cm, annot=True,annot_kws={"size": 16},fmt='g')# font size

'''基于oneclasssvm的异常检测'''
alg=OneClassSVM(kernel='linear',gamma='auto', coef0=0.0, tol=0.001, nu=0.5,
                shrinking=True, cache_size=500, verbose=False, max_iter=-1)

osvm_mdlLst=train(X_train,alg)
osvm_y_pred=predict(X_test,osvm_mdlLst)

#Creating class labels based on decision function
osvm_y_pred_class=osvm_y_pred.copy()
osvm_y_pred_class[osvm_y_pred<0]=1
osvm_y_pred_class[osvm_y_pred>=0]=0


roc_auc_score(y_test, osvm_y_pred_class)
f1_score(y_test, osvm_y_pred_class)
osvm_cm=confusion_matrix(y_test, osvm_y_pred_class)


df_cm = pd.DataFrame(osvm_cm,['True Normal','True Fraud'],['Pred Normal','Pred Fraud'])
pyplot.figure(figsize = (8,4))
sn.set(font_scale=1.4) #for label size
sn.heatmap(df_cm, annot=True,annot_kws={"size": 16},fmt='g')# font size


# pyplot.title('Training Time')
# pyplot.barh(range(len(train_times)), list(train_times.values()), align='center')
# pyplot.yticks(range(len(train_times)), list(train_times.keys()))
# pyplot.xlabel('Time in seconds')


auc_scores={
    'Isolation Forest': roc_auc_score(y_test, if_y_pred_class),
    'KMeans':roc_auc_score(y_test, km_y_pred),
    'LOF':roc_auc_score(y_test, lof_y_pred_class),
    'OneClass SVM': roc_auc_score(y_test, osvm_y_pred_class)
}

f1_scores={
    'Isolation Forest':f1_score(y_test, if_y_pred_class),
    'KMeans':f1_score(y_test, km_y_pred),
    'LOF':f1_score(y_test, lof_y_pred_class),
    'OneClass SVM': f1_score(y_test, osvm_y_pred_class)
}


print('auc data',auc_scores)
print('f1 score',f1_scores)
# pyplot.title('AUC Scores')
# pyplot.barh(range(len(auc_scores)), list(auc_scores.values()), align='center')
# pyplot.yticks(range(len(auc_scores)), list(auc_scores.keys()))
# pyplot.xlabel('AUC Score')
# pyplot.show()


# pyplot.title('F1 Scores')
# pyplot.barh(range(len(f1_scores)), list(f1_scores.values()), align='center')
# pyplot.yticks(range(len(f1_scores)), list(f1_scores.keys()))
# pyplot.xlabel('F1 Score')
# pyplot.show()





