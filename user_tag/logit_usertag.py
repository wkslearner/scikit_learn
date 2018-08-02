import pandas as pd
import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier
from statsmodels.stats.outliers_influence import variance_inflation_factor
from score_card_version1.woe_information import *
import statsmodels.api as sm
from score_card_version1.result_check import KS_AR
from sklearn.metrics import roc_curve,auc,roc_auc_score
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

#scipy中ks函数
from scipy.stats import ks_2samp

start_time=time.time()
user_df=pd.read_csv('/Users/andpay/Desktop/portrait.csv')
var_list=['register','sumamt','applycount','age','cardnum','count_d','use_rate','sex']


#'passcount','loancount','lineused','pass_rate','withdraw_rate',,'counts',
# 'cardnum','count_d','use_rate','sex'


'''随机森林变量重要性排序'''
x=user_df[var_list]
y=user_df['cate']

RFC = RandomForestClassifier()
RFC.fit(x,y.astype(int))
features_rfc = var_list
featureImportance = {features_rfc[i]:RFC.feature_importances_[i] for i in range(len(features_rfc))} #变量重要性字典
featureImportanceSorted = sorted(featureImportance.items(),key=lambda x: x[1], reverse=True) #按照重要性排序
# we selecte the top 10 features
features_selection = [k[0] for k in featureImportanceSorted] #获取排名靠前的12个特征变量
print(features_selection)


'''变量间相关性检验'''
var_relative=regression_analysis(user_df,var_list)
print(var_relative)


'''
多变量共线性检测：VIF（方差膨胀因子）
'''

matx = np.matrix(user_df[var_list])
for i in range(len(var_list)):
    vif=variance_inflation_factor(matx,i)
    print(var_list[i],vif)


traindata, testdata= train_test_split(user_df,test_size=0.2)

x_train,y_train=traindata[var_list],traindata['cate']
x_test,y_test=testdata[var_list],testdata['cate']

user_df['intercept'] = [1]*user_df.shape[0]
LR = sm.Logit(y_train,x_train).fit()
summary = LR.summary()
user_df['prob']=LR.predict(user_df[var_list])
ks_dict=KS_AR(user_df,'prob','cate')
KS=ks_dict['KS']
auc = roc_auc_score(user_df['cate'].astype(int),user_df['prob'])

print(summary)
print(KS,auc)

#逻辑回归结果精度验证
y_train_pred = LR.predict(x_train)
y_test_pred = LR.predict(x_test)
tree_train = accuracy_score(y_train, y_train_pred)
tree_test = accuracy_score(y_test.astype(int), y_test_pred.astype(int))
print('Random Forest train/test accuracies %.3f/%.3f' % (tree_train, tree_test))


'''
excel_writer=pd.ExcelWriter('/Users/andpay/Desktop/portrait_mode.xlsx',engine='xlsxwriter')
user_df.to_excel(excel_writer,'mode_result',index=False)
excel_writer.save()
'''

end_time=time.time()
print(end_time-start_time)

