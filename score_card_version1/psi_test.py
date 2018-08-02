import  pandas as pd
from score_card_version1.split_bin import *
from score_card_version1.preprocessing import *
from score_card_version1.woe_information import *
from numpy import *
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve,auc,roc_auc_score
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.ensemble import RandomForestClassifier
import statsmodels.api as sm
from score_card_version1.result_check import *
from sklearn.cross_validation import train_test_split



df=pd.read_excel('/Users/andpay/Documents/job/mode/12.25score.xlsx')
df=df.drop(['PARTYID','IDNO','TDID','MAXLD'],axis=1)
df.loc[df['CATEGORY']=='NM','CATEGORY']=0
df.loc[df['CATEGORY']=='M2','CATEGORY']=1


'''去除空值大于一半的列'''
unuse_list=check_nullvalue(df)
df=df.drop(unuse_list,axis=1)


col=list(df.columns)
col.remove('CATEGORY')

cat_var,con_var=category_var(df[col])
split_point_cat,disper_df=disper_split(df,cat_var)
split_point_chi,chi_df=chi_equalwide(df,con_var,'CATEGORY',max_interval=5,mont=True,special_list=['AGE','TD_SCORE'])

new_df=pd.concat([disper_df,chi_df,df['CATEGORY']],axis=1)
new_df['intercept']=[1]*new_df.shape[0]
x_train, x_test= train_test_split(new_df,test_size=0.2) #random_state=new_df.shape[0]


end_var_list =['ZMS_freq_bin','TD_SCORE_freq_bin','USERATE_freq_bin',#'LASTDATE_freq_bin',
              'ACS_9_freq_bin',#'TLBALANCE_freq_bin',#'TCL_freq_bin','HLWCX_freq_bin',
              'AGE_freq_bin','SUMAMT_freq_bin','MINDATE_freq_bin','intercept']


'''变量间相关性检验'''
var_relative=regression_analysis(x_train,end_var_list)
print(var_relative)


'''
多变量共线性检测：VIF（方差膨胀因子）
'''
matx = np.matrix(new_df[end_var_list])
for i in range(len(end_var_list)):
    vif=variance_inflation_factor(matx,i)
    print(end_var_list[i],vif)


#print(x_train[0:5])


'''建立模型'''
LR_train=sm.Logit(np.array(x_train['CATEGORY']).astype(int),np.array(x_train[end_var_list])).fit()


'''
#采用L1正则化约束模型，通过迭代alpha取值
for alpha in range(100,0,-1):
    l1_logit = sm.Logit.fit_regularized(sm.Logit(np.array(x_train['CATEGORY']).astype(int),np.array(x_train[end_var_list])),
                                        start_params=None, method='l1', alpha=alpha)
    pvalues = l1_logit.pvalues
    params = l1_logit.params
    if max(pvalues)>=0.1 or max(params)>0:
        break

bestAlpha = alpha + 1
LR_train = sm.Logit.fit_regularized(sm.Logit(np.array(x_train['CATEGORY']).astype(int),np.array(x_train[end_var_list])),
                                    start_params=None, method='l1', alpha=bestAlpha)

params = l1_logit.params
'''

summary = LR_train.summary()

''' 模型KS值，PSI值、AUC值及描述信息 '''
x_train['prob']=LR_train.predict(sm.add_constant(x_train[end_var_list]))
ks_dict=KS_AR(x_train,'prob','CATEGORY')
KS=ks_dict['KS']
aucs= roc_auc_score(x_train['CATEGORY'].astype(int),x_train['prob'])
print(KS,aucs)
print(summary)


x_test['prob']=LR_train.predict(x_test[end_var_list])


psi=PSI(x_train,x_test,'prob','prob')
print(psi)

para=LR_train.params
mode_woe_df,mode_information_df=get_woe_information(new_df,end_var_list,'CATEGORY')

#合并两个标签字典
combine_point_dict=dict(split_point_cat,**split_point_chi)

#创建截距字典
coef_dict={}
for key,value in zip(end_var_list,para):
    coef_dict[key]=value

#把截距和斜率值放入woe的结果集中
for var,key in zip(mode_woe_df['variable'],mode_woe_df['class']):
    if var=='intercept':
        continue
    else:
        mode_woe_df.loc[(mode_woe_df['variable']==var)&(mode_woe_df['class']==key),'category']= combine_point_dict[var][key]


for var in mode_woe_df['variable'].unique():
    mode_woe_df.loc[(mode_woe_df['variable']==var),'coef']=coef_dict[var]


print(mode_information_df.sort_values(by='information_value',ascending=[0]))
print(mode_woe_df)



'''
excel_writer=pd.ExcelWriter('/Users/andpay/Documents/job/mode/xx.xlsx',engine='xlsxwriter')
mode_woe_df.to_excel(excel_writer,'mode_woe',index=False)
mode_information_df.to_excel(excel_writer,'informationvalue',index=False)
#monotone_df.to_excel(excel_writer,'monotone',index=False)
excel_writer.save()
'''

