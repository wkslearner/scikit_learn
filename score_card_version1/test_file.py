'''
1、特征处理
2、特征分箱(先等频后卡方，同时检验数据的单调性)
3、计算woe和iv值，并根据iv值挑选可用变量
4、单变量相关性检验，多变量共线性检测
5、用变量进行建模
6、剔除显著性较低变量
'''

import numpy as np
import  pandas as pd
from score_card_version1.split_bin import *
from score_card_version1.preprocessing import *
from score_card_version1.woe_information import *
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve,auc,roc_auc_score
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.ensemble import RandomForestClassifier
from xgboost.sklearn import XGBClassifier
import statsmodels.api as sm
from score_card_version1.result_check import *
from sklearn.cross_validation import train_test_split
from data_process.list_process import remove_list
from plot_function.model_plot import plot_KS,plot_ROC
from data_sample.random_sample import RandomSample
from sklearn import metrics
from sklearn.model_selection import learning_curve
from sklearn import preprocessing
from imblearn.over_sampling import SMOTE


df=pd.read_excel('/Users/andpay/Documents/job/model/scorecard_model/scorecard_model_train_v2.xlsx')
end_user_info=df.drop(['partyid','loanid','loantime','min_time'],axis=1)
end_user_info.loc[end_user_info['over_dueday']>=30,'cate']=1
end_user_info.loc[end_user_info['over_dueday']<30,'cate']=0
end_user_info['cardline_userate']=end_user_info['cardlineused']/end_user_info['cardline']
end_user_info['cardline_userate_3m']=end_user_info['cardlineused_3m']/end_user_info['cardline_3m']

print(end_user_info['over_dueday'].groupby(end_user_info['cate']).count())

var_list=['txn_avgamt_6m','txn_count_6m','query_num_3m','cardline','cardline_3m','pcr_loanamt','pcr_loanamt_3m',
          'pcr_other_balance','pcr_over_otheramt','pcr_over_amt','td_score','p2p_1m','p2p_3m','p2p_7d','platform_count_1m',
          'platform_count_3m','platform_count_7d','normal_consume_1m','normal_consume_3m','normal_consume_7d','net_finace_1m',
          'net_finace_3m','net_finace_7d','credit_centen_1m','credit_centen_3m','credit_centen_7d','big_consume_1m',
          'big_consume_3m','big_consume_7d','small_loan_1m','small_loan_3m','small_loan_7d','bank_consume_1m','bank_consume_3m',
          'bank_consume_7d']


column=end_user_info.columns
print(end_user_info.shape)


# 对好样本进行随机欠采样
# random_sample=RandomSample(np.array(end_user_info[end_user_info['cate']==0]),0.3).random_under_sample()
# random_sample_df=pd.DataFrame(random_sample,columns=end_user_info.columns)
# end_user_info=pd.concat([end_user_info[end_user_info['cate']==1],random_sample_df],axis=0)
# print(end_user_info.shape)


'''缺失值检测'''
unuse_list=check_nullvalue(end_user_info)
end_user_info=end_user_info.drop(unuse_list,axis=1)
new_column=end_user_info.columns


cat_var=['id_city']
con_var=remove_list(new_column,cat_var+['over_dueday','cate'])
end_user_info=disper_split(end_user_info,cat_var)



'''变量间相关性检验'''
var_relative=regression_analysis(end_user_info,con_var)
relative_df=pd.DataFrame(var_relative,columns=['var1','var2','p_value','relative_coeffient'])
print(relative_df)


#对变量进行卡方分箱
split_point_chi,chi_df_1=chi_equalwide(end_user_info,con_var,'cate',max_interval=5,numOfSplit=300,mont=False,special_list=['age'])
end_col=list(chi_df_1.columns)



'''卡方分箱后手动合并分箱'''
def manual_mergebin(dataframe,merge_var,merge_bin=[],split_point_dict={}):

    merge_value=min(merge_bin)
    dataframe.loc[dataframe[merge_var].isin(merge_bin),merge_var]=merge_value

    split_dict=split_point_dict[merge_var]

    bin_list=[]
    for key in merge_bin:
        bin=split_dict[key].split('-')
        bin_list=bin_list+bin

        #删除字典中原有项
        split_dict.pop(key)

    #生成新键值对
    bin_text=str(bin_list[0])+'-'+str(bin_list[-1])
    #把新键值对放入字典
    split_dict[merge_value]=bin_text
    #重新对原因变量标签进行赋值
    split_point_dict[merge_var]=split_dict

    return dataframe,split_point_dict


end_user_info,split_point_chi=manual_mergebin(end_user_info,'td_score_freq_bin',merge_bin=[2,3,4],split_point_dict=split_point_chi)
end_user_info,split_point_chi=manual_mergebin(end_user_info,'txn_count_6m_freq_bin',merge_bin=[2,3],split_point_dict=split_point_chi)
print(split_point_chi)


woe_df,information_df=get_woe_information(end_user_info,end_col,'cate')
for var,key in zip(woe_df['variable'],woe_df['class']):
    woe_df.loc[(woe_df['variable']==var)&(woe_df['class']==key),'category']= split_point_chi[var][key]


# print(woe_df[woe_df['variable'].isin(['USERATE_freq_bin','TD_SCORE_freq_bin'])])

# excel_writer=pd.ExcelWriter('/Users/andpay/Documents/job/model/scorecard_model/woe_iv.xlsx',engine='xlsxwriter')
# woe_df.to_excel(excel_writer,'woe',index=False)
# information_df.to_excel(excel_writer,'informationvalue',index=False)
# excel_writer.save()


#smote算法进行数据过采样
# over_sample=SMOTE(ratio=0.8)
# x,y=over_sample.fit_sample(end_user_info[end_col],end_user_info['cate'])


'''随机森林挑选变量'''
x=end_user_info[end_col]
y=end_user_info['cate']


RFC =XGBClassifier()
RFC.fit(x,y.astype(int))
xgc_col=list(np.round(RFC.feature_importances_,3))


var_importance=pd.DataFrame({'var':end_col,'importance_value':xgc_col})
var_importance=var_importance.sort_values(by='importance_value',ascending=0)
print(var_importance)


# features_rfc = end_col
# featureImportance = {features_rfc[i]:RFC.feature_importances_[i] for i in range(len(features_rfc))} #变量重要性字典
# featureImportanceSorted = sorted(featureImportance.items(),key=lambda x: x[1], reverse=True) #按照重要性排序
# # we selecte the top 10 features
# features_selection = [k[0] for k in featureImportanceSorted] #获取排名靠前的12个特征变量
# print(features_selection)

end_user_info=pd.DataFrame(x,columns=end_col)
end_user_info['cate']=y

#移除方差膨胀因子高的变量
end_col=remove_list(end_col,['query_num_3m_freq_bin','cardline_freq_bin','cardline_3m_freq_bin','cardlineused_3m_freq_bin','cardlineused_freq_bin',
                             'pcr_loanamt_freq_bin','cardline_userate_3m_freq_bin','p2p_1m_freq_bin','platform_count_1m_freq_bin',
                             'pcr_other_balance_freq_bin','platform_count_3m_freq_bin'])



'''
多变量共线性检测：VIF（方差膨胀因子）
'''
matx = np.matrix(end_user_info[end_col])
for i in range(len(end_col)):
    vif=variance_inflation_factor(matx,i)
    print(end_col[i],vif)



def vif(matx):
    vif_list=[]
    for i in range(matx.shape[1]):
        vif=variance_inflation_factor(matx,i)
        vif_list.append(vif)

    max_vif=max(vif_list)

    return max_vif


'''
for i in range(len(end_var_list)):
    new_list = end_var_list.copy()
    print(new_list[i])
    new_list.remove(new_list[i])
    matx=np.matrix(end_user_info[new_list])
    vifs=vif(matx)

    print(vifs)
'''

locgit=LogisticRegression(fit_intercept = False,C = 1e9)
'''
locgit=LogisticRegression(fit_intercept = False,C = 1e9)
locgit.fit(the_end_df[end_var_list],the_end_df['CATEGORY'].astype(int))
intercept=locgit.intercept_
coef=locgit.coef_
the_end_df['prob_1']=locgit.predict_proba(the_end_df[end_var_list])[:,1]
ks_dict_1=KS_AR(the_end_df,'prob_1','CATEGORY')
KS_1=ks_dict_1['KS']
print(coef,intercept,KS_1)
'''


'''逻辑回归'''
end_user_info['intercept'] = [1]*end_user_info.shape[0]
LR = sm.Logit(end_user_info['cate'].astype(int),end_user_info[end_col]).fit()
summary = LR.summary()
end_user_info['prob']=LR.predict(end_user_info[end_col])
ks_dict=KS_AR(end_user_info,'prob','cate')
KS=ks_dict['KS']
auc_result = roc_auc_score(end_user_info['cate'].astype(int),end_user_info['prob'])

y_train_pred_xgb = LR.predict(end_user_info[end_col])
#.apply(lambda x: 1 if x>=0.5 else 0)
# tree_train = metrics.accuracy_score(end_user_info['cate'].astype(int), y_train_pred_xgb)
# print('XG Boosting train/test accuracies %.3f' % tree_train)


print('XG Boosting train/test auc %.3f '%(metrics.roc_auc_score(end_user_info['cate'].astype(int),y_train_pred_xgb)))
# print('XG Boosting train/test Recall %.3f' % (metrics.recall_score(end_user_info['cate'].astype(int),y_train_pred_xgb)))
# print('XG Boosting train/test precision %.3f' % (metrics.precision_score(end_user_info['cate'].astype(int),y_train_pred_xgb)))


#画KS曲线信息
# ks_plot=plot_KS(end_user_info['prob'],end_user_info['cate'],10000,asc=0)


basePoint = 600
PDO = 20
end_user_info['score'] = end_user_info['prob'].map(lambda x:Prob2Score(x, basePoint, PDO))
#testData = the_end_df.sort_values(by = 'score')


print(summary)
print(KS,auc_result)


'''查看分数分布图'''
'''
plt.hist(the_end_df['score'], 100)
plt.xlabel('score')
plt.ylabel('freq')
plt.title('distribution')
plt.show()
'''

mode_woe_df,mode_information_df=get_woe_information(end_user_info,end_col,'cate')


'''画ROC曲线'''
#probs=LR.predict_proba(end_user_info[end_col])

# fpr,tpr,threshold=roc_curve(end_user_info['cate'].astype(int),end_user_info['prob'])
# roc_auc = auc(fpr, tpr)

# plt.title('Receiver Operating Characteristic')
# plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
# plt.legend(loc = 'lower right')
# plt.plot([0, 1], [0, 1],'r--')
# plt.xlim([0, 1])
# plt.ylim([0, 1])
# plt.ylabel('True Positive Rate')
# plt.xlabel('False Positive Rate')
# plt.show()



'''
#创建截距字典
coef_dict={}
for key,value in zip(end_var_list,coef[0]):
    coef_dict[key]=value


#把截距和斜率值放入woe的结果集中
for var,key in zip(mode_woe_df['variable'],mode_woe_df['class']):
    mode_woe_df.loc[(mode_woe_df['variable']==var)&(mode_woe_df['class']==key),'category']= combine_point_dict[var][key]

for var in mode_woe_df['variable'].unique():
    mode_woe_df.loc[(mode_woe_df['variable']==var),'coef']=coef_dict[var]

mode_woe_df['intercept']=intercept


excel_writer=pd.ExcelWriter('/Users/andpay/Documents/job/mode/xxx.xlsx',engine='xlsxwriter')
woe_df.to_excel(excel_writer,'woe',index=False)
information_df.to_excel(excel_writer,'information',index=False)
mode_woe_df.to_excel(excel_writer,'mode_woe',index=False)
mode_information_df.to_excel(excel_writer,'informationvalue',index=False)
#monotone_df.to_excel(excel_writer,'monotone',index=False)
excel_writer.save()
'''



