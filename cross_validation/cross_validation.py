#!/usr/bin/python
# encoding=utf-8
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn import cross_validation
from sklearn.datasets import load_sample_image


#获取波士顿房价数据集
boston=load_boston()
value=boston.data
feature=boston.feature_names
target=boston.target

boston_df=pd.DataFrame(value,columns=feature)
boston_df['target']=target
boston_df=boston_df.fillna(0)
boston_df.loc[boston_df['target']<=25,'target']=0
boston_df.loc[boston_df['target']>25,'target']=1

x_variable=np.array(boston_df[feature])
y_variable=np.array(boston_df['target'])


'''k折交叉验证'''
kf=KFold(n_splits=10)

acc_list = []
for train_index,test_index in kf.split(x_variable):#获取分割数据集索引

    #对自变量和目标变量数据进行分割
    x_train,x_test=x_variable[train_index],x_variable[test_index]
    y_train,y_test=y_variable[train_index],y_variable[test_index]

    #建立随机森林
    model=RandomForestClassifier(n_estimators=100)
    #训练模型
    model.fit(x_train,y_train)

    train_pred=model.predict(x_train)
    test_pred=model.predict(x_test)

    test_acc=metrics.accuracy_score(y_test,test_pred)
    acc_list.append(test_acc)

    #精确率，准确率，召回率
    print('acc',metrics.accuracy_score(y_train,train_pred),metrics.accuracy_score(y_test,test_pred))
    print('precision',metrics.precision_score(y_train, train_pred), metrics.precision_score(y_test, test_pred))
    print('recall',metrics.recall_score(y_train, train_pred), metrics.recall_score(y_test, test_pred))
    print(' ')

print(np.mean(acc_list))



