import  pandas as pd
from score_card_version1.split_bin import *
from score_card_version1.preprocessing import *
from score_card_version1.woe_information import *
from numpy import *
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve,auc
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.ensemble import RandomForestClassifier
from score_card_version1.test_file import chi_equalwide,disper_split

df=pd.read_excel('/Users/andpay/Documents/job/mode/score_card12.7.xlsx')
end_user_info=df.drop(['PARTYID','NAME','IDNO'],axis=1)
end_user_info.loc[end_user_info['CATEGORY']=='NM','CATEGORY']=0
end_user_info.loc[end_user_info['CATEGORY']=='M2','CATEGORY']=1


column=end_user_info.columns

'''根据空值情况筛选变量'''
use_list=[]
unuse_list=[]
for col in column:
    res=check_nullvalue(end_user_info,col)
    if res=='unuseable':
        unuse_list.append(col)
    else:
        use_list.append(col)

end_user_info=end_user_info.drop(unuse_list,axis=1)
new_column=end_user_info.columns




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



disper_split(end_user_info)
chi_equalwide(end_user_info,con_var,'CATEGORY',mont=True,special_list=['AGE_freq'])
