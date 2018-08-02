'''
1、特征处理
2、特征分箱(先等频后卡方，同时检验数据的单调性)
3、计算woe和iv值，并根据iv值挑选可用变量
4、单变量相关性检验，多变量共线性检测
5、用变量进行建模
6、剔除显著性较低变量
'''


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
end_user_info=df.drop(['PARTYID','IDNO','TDID','MAXLD'],axis=1)
end_user_info.loc[end_user_info['CATEGORY']=='NM','CATEGORY']=0
end_user_info.loc[end_user_info['CATEGORY']=='M2','CATEGORY']=1


column=end_user_info.columns

'''缺失值检测'''
unuse_list=check_nullvalue(end_user_info)
end_user_info=end_user_info.drop(unuse_list,axis=1)
new_column=end_user_info.columns

print(new_column)


'''把变量非空取值数小余5的归为分类变量,大于5的为连续变量'''
def category_var(dataframe):
    cat_list=[]
    con_list=[]
    column=dataframe.columns
    for var in column:
        value_list=dataframe[dataframe[var].notnull()][var].unique()
        if len(value_list)<=5:
            cat_list.append(var)
        else:
            con_list.append(var)

    return cat_list,con_list


cat_var,con_var=category_var(end_user_info)
cat_var.remove('CATEGORY')
con_var_1=['3M_HLWCX','TLBALANCE','TCL','USERATE','BAO','HLWCX','MINDATE']
con_var_2=con_var

for key in con_var_1:
    con_var_2.remove(key)


def disper_split(dataframe):
    '''分类变量处理'''
    split_point_cat={}
    split_cat_list = []
    for var in cat_var:
        split_cat_list.append(var + '_cat')
        mid_dict={}
        if dataframe[dataframe[var].isnull()].shape[0] > 0:
            sort_value = sorted(list(dataframe[dataframe[var].notnull()][var].unique()))
            num = len(sort_value)
            for i in range(num):
                dataframe.loc[(dataframe[var] == sort_value[i]), var + '_cat'] = i
                mid_dict[i]=sort_value[i]

            dataframe.loc[dataframe[var].isnull(), var + '_cat'] = -1
            mid_dict[-1]='None'
            split_point_cat[var+'_cat']=mid_dict

        else:
            sort_value = sorted(list(dataframe[dataframe[var].notnull()][var].unique()))
            num = len(sort_value)
            for i in range(num):
                dataframe.loc[(dataframe[var] == sort_value[i]), var + '_cat'] = i
                mid_dict[i] = sort_value[i]

            split_point_cat[var + '_cat'] = mid_dict

    return split_point_cat,dataframe[split_cat_list]


split_point_cat,disper_df=disper_split(end_user_info)

def chi_equalwide(dataframe,var_list,target_var,mont=True,numOfSplit=200,max_interval=5,special_list=[]):
    '''连续变量进行等频分箱'''
    split_point_freq = {}
    freq_var_list = []
    for var in var_list:
        freq_var_list.append(var + '_freq')
        split_point = basic_splitbin(dataframe, var, numOfSplit=numOfSplit)  # 先对数据进行等频分箱，分为100箱
        split_point_freq[var + '_freq'] = split_point
        num = len(split_point)

        for i in range(num - 1):
            before = split_point[i]
            after = split_point[i + 1]
            # 分箱后新设变量并重新赋值
            dataframe.loc[(dataframe[var] >= before) & (dataframe[var] < after), var + '_freq'] = i
            dataframe.loc[dataframe[var] >= split_point[num - 1], var + '_freq'] = i + 1  # 最后一个分箱单独处理

    '''构造特殊变量'''
    new_special_list = []
    for var in special_list:
        new_special_list.append(var + '_freq')


    '''等频变量卡方分箱'''
    split_point_chi={}
    chi_var_list=[]
    monotone_list=[]
    for var in freq_var_list:
        '''首次对数据分为5箱'''
        chi_var_list.append(var+'_bin')
        split_point=ChiMerge(dataframe,var,target_var,max_interval=max_interval)  #卡方分箱，返回分箱分割点
        freq_value=split_point_freq[var] #引入等频分箱的数据分割点列表

        num = len(split_point)
        mid_dict = {}

        for i in range(num-1):
            before=split_point[i]
            after=split_point[i+1]
            min_value=freq_value[int(before)]
            max_value=freq_value[int(after)]
            bin_name=str(min_value) + '-' + str(max_value)
            #分箱后新设变量并重新赋值
            dataframe.loc[(dataframe[var] >= before) & (dataframe[var] < after), var+'_bin'] = i
            mid_dict[i] =bin_name  #保存分箱范围到字典

        #最后一个分箱处理
        dataframe.loc[dataframe[var] >= split_point[num - 1], var+'_bin'] = i + 1
        specil_value=freq_value[int(split_point[num - 1])]
        mid_dict[i+1]=str(specil_value)+'-'+'inf'

        '''单调性检验,并做重新分箱处理'''
        mot = monotone(dataframe, var + '_bin', target_var)  # 单调性检验

        if mont==True:
            '''如果存在其他业务上可理解的非单调性变量（如年龄），需要申明special_list'''
            if var not in new_special_list:
                while (not mot):
                    max_interval -= 1
                    split_point = ChiMerge(dataframe, var, target_var, max_interval=max_interval)
                    num = len(split_point)
                    mid_dict = {}

                    for i in range(num - 1):
                        before = split_point[i]
                        after = split_point[i + 1]
                        min_value = freq_value[int(before)]
                        max_value = freq_value[int(after)]
                        bin_name = str(min_value) + '-' + str(max_value)
                        # 分箱后新设变量并重新赋值
                        dataframe.loc[(dataframe[var] >= before) & (dataframe[var] < after), var + '_bin'] = i
                        mid_dict[i] = bin_name  #保存分箱范围到字典

                    dataframe.loc[dataframe[var] >= split_point[num - 1], var + '_bin'] = i + 1
                    specil_value = freq_value[int(split_point[num - 1])]
                    mid_dict[i + 1] = str(specil_value) + '-' + 'inf'

                    mot = monotone(dataframe, var+'_bin', target_var)

        monotone_list.append([var+'_bin',mot])
        dataframe.loc[dataframe[var].isnull(),var+'_bin']=-1  #空值分箱单独处理
        mid_dict[-1]=['None']
        split_point_chi[var+'_bin']=mid_dict

    return split_point_chi,dataframe[chi_var_list]


split_point_chi_1,chi_df_1=chi_equalwide(end_user_info,con_var_2,'CATEGORY',max_interval=5,mont=False,special_list=['AGE'])
split_point_chi_2,chi_df_2=chi_equalwide(end_user_info,con_var_1,'CATEGORY',max_interval=5,mont=True,special_list=['AGE'])

combine_var_list=list(disper_df.columns)+list(chi_df_1.columns)+list(chi_df_2)+['CATEGORY']
combine_point_dict=dict(split_point_cat,**split_point_chi_1)
combine_point_dict=dict(combine_point_dict,**split_point_chi_2)
the_end_df=pd.concat([disper_df,chi_df_1,chi_df_2,end_user_info['CATEGORY']],axis=1) #处理后的分类变量，连续变量与目标变量整合


'''woe和iv值计算'''
def get_woe_information(dataframe,variable_list,target_var):
    #求woe 和 infomation_value
    woe_dict={}
    information_list=[]
    for var in variable_list:
        woe,information_value=woe_informationvalue(dataframe,var,target_var)
        woe_dict[var]=woe
        information_list.append(information_value)


    #把所有值进行分解并封装成dataframe
    woe_list = []
    for var in variable_list:
        first_layer = woe_dict[var]
        for key in first_layer.keys():
            value = first_layer[key]['woe']
            count = first_layer[key]['count']
            woe_list.append([var, key, count,value])


    woe_df = pd.DataFrame(woe_list, columns=['variable', 'class','bad_count','woe'])
    information_df = pd.DataFrame({'variable': variable_list, 'information_value': information_list},
                                  columns=['variable', 'information_value'])

    woe_df=woe_df[woe_df['variable']!=target_var]  #结果中不呈现目标变量
    information_df=information_df[information_df['variable']!=target_var]

    return woe_df,information_df


woe_df,information_df=get_woe_information(the_end_df,combine_var_list,'CATEGORY')

for var,key in zip(woe_df['variable'],woe_df['class']):
    woe_df.loc[(woe_df['variable']==var)&(woe_df['class']==key),'category']= combine_point_dict[var][key]

#print(woe_df[woe_df['variable'].isin(['USERATE_freq_bin','TD_SCORE_freq_bin'])])


'''
excel_writer=pd.ExcelWriter('/Users/andpay/Documents/job/mode/xxx.xlsx',engine='xlsxwriter')
woe_df.to_excel(excel_writer,'woe',index=False)
information_df.to_excel(excel_writer,'informationvalue',index=False)
excel_writer.save()
'''



'''
end_var_list=['LASTDATE_freq_bin','ZMS_freq_bin','TD_SCORE_freq_bin','MONCOUNT_freq_bin','SUMAMT_freq_bin','COUNTS_freq_bin',
              'ACS_9_freq_bin','LOANTIMES_freq_bin','USC_9_freq_bin','FISTDATE_freq_bin','LOANAMT_freq_bin','CCNUM_freq_bin',
              'ACS_0_freq_bin','TLBALANCE_freq_bin','USC_0_freq_bin','TCL_freq_bin','TCLU_freq_bin','USERATE_freq_bin','AGE_freq_bin',
              'LG_cat']

'''


end_var_list=['ZMS_freq_bin','TD_SCORE_freq_bin','USERATE_freq_bin','LASTDATE_freq_bin',
              'ACS_9_freq_bin','TLBALANCE_freq_bin','LOANAMT_freq_bin','TCL_freq_bin',
              'AGE_freq_bin','SUMAMT_freq_bin','HLWCX_freq_bin','MINDATE_freq_bin','LG_freq_bin'
              ]



'''随机森林挑选变量'''

x=end_user_info[end_var_list]
y=end_user_info['CATEGORY']

RFC = RandomForestClassifier()
RFC.fit(x,y.astype(int))
features_rfc = end_var_list
featureImportance = {features_rfc[i]:RFC.feature_importances_[i] for i in range(len(features_rfc))} #变量重要性字典
featureImportanceSorted = sorted(featureImportance.items(),key=lambda x: x[1], reverse=True) #按照重要性排序
# we selecte the top 10 features
features_selection = [k[0] for k in featureImportanceSorted] #获取排名靠前的12个特征变量
print(features_selection)


'''变量间相关性检验'''
var_relative=regression_analysis(the_end_df,end_var_list)
print(var_relative)


'''
多变量共线性检测：VIF（方差膨胀因子）
'''
matx = np.matrix(the_end_df[end_var_list])
for i in range(len(end_var_list)):
    vif=variance_inflation_factor(matx,i)
    print(end_var_list[i],vif)



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


#print(x_train,x_test)


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
the_end_df['intercept'] = [1]*the_end_df.shape[0]
LR = sm.Logit(the_end_df['CATEGORY'].astype(int),the_end_df[end_var_list]).fit()
summary = LR.summary()
the_end_df['prob']=LR.predict(the_end_df[end_var_list])
ks_dict=KS_AR(the_end_df,'prob','CATEGORY')
KS=ks_dict['KS']
auc = roc_auc_score(the_end_df['CATEGORY'].astype(int),the_end_df['prob'])



basePoint = 600
PDO = 20
the_end_df['score'] = the_end_df['prob'].map(lambda x:Prob2Score(x, basePoint, PDO))
#testData = the_end_df.sort_values(by = 'score')


print(summary)
print(KS,auc)


'''查看分数分布图'''
'''
plt.hist(the_end_df['score'], 100)
plt.xlabel('score')
plt.ylabel('freq')
plt.title('distribution')
plt.show()
'''

mode_woe_df,mode_information_df=get_woe_information(the_end_df,end_var_list,'CATEGORY')


'''画ROC曲线'''

'''
probs=LR.predict_proba(end_user_info[end_var_list])

fpr,tpr,threshold=roc_curve(end_user_info['CATEGORY'].astype(int),probs[:,1])
roc_auc = auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
'''


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



