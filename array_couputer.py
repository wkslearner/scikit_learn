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


user_df=pd.read_csv('/Users/andpay/Desktop/portrait.csv')

var_list=['register','sumamt','applycount','age', 'cardnum','count_d','use_rate','sex',
          'cardnum','count_d','passcount','loancount','lineused','counts']


traindata, testdata= train_test_split(user_df,test_size=0.2)


x_train,y_train=traindata[var_list],traindata['cate']
x_test,y_test=testdata[var_list],testdata['cate']


for i in range(10,100,5):
    clf = GradientBoostingClassifier(n_estimators=i,learning_rate=0.05,max_depth=2,random_state=0)
    clf.fit(x_train, y_train.astype(int))
    print ("n_estimators = "+str(i)+"  learning_rate = "+str(0.05)+ \
    "  score = "+str(clf.score(x_test, y_test.astype(int))))

