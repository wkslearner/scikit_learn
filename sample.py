
import random
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split
import information_woe  as iw
from sklearn.linear_model import LogisticRegression
from line_mode import chimerge
from sklearn.metrics import roc_curve,auc
import matplotlib.pyplot as plt

'''使用mongo数据进行处理'''

"""
user_info=pd.read_excel('/Users/andpay/Documents/job/mode/random_applyid_data.xlsx')
m2_df=pd.read_csv('/Users/andpay/Documents/job/mode/M2_list.csv')

end_user_info=pd.merge(user_info,m2_df,left_on='partyid',right_on='PARTYID')
end_user_info=end_user_info.drop(['partyid','applyid','phone','PARTYID','city'],axis=1)
end_user_info.loc[end_user_info['CATEGROY']=='NM','CATEGROY']=0
end_user_info.loc[end_user_info['CATEGROY']=='M2','CATEGROY']=1

"""

user_info=pd.read_excel('/Users/andpay/Documents/job/mode/score_card12.7.xlsx')
end_user_info=user_info.drop(['PARTYID','NAME','IDNO'],axis=1)
end_user_info.loc[end_user_info['CATEGORY']=='NM','CATEGORY']=0
end_user_info.loc[end_user_info['CATEGORY']=='M2','CATEGORY']=1

column=end_user_info.columns

#根据空值情况筛选变量
use_list=[]
unuse_list=[]
for col in column:
    res=iw.check_nullvalue(end_user_info,col)
    if res=='unuseable':
        unuse_list.append(col)
    else:
        use_list.append(col)

end_user_info=end_user_info.drop(unuse_list,axis=1)

new_column=end_user_info.columns

#检测变量间的共线性
#var_relative=iw.regression_analysis(end_user_info,new_column)
print('end first step')

def variable_handle(dataframe,target_variable,method,max_interval):
    coutinue_variable = list(dataframe.columns)
    coutinue_variable.remove(target_variable)

    #判断是否为连续变量
    continue_variable_list=[]
    category_variable_list=[]
    for key_word in coutinue_variable:
        '''取值个数大于10个以上的为连续变量'''
        if len(dataframe[key_word].unique())>10:
            continue_variable_list.append(key_word)
        else:
            category_variable_list.append(key_word)

    continue_point_dict={}
    category_point_dict={}

    if 'equal_wide' in method :
        '''对连续变量进行等宽分段处理'''
        for column_name in continue_variable_list:
            split_point=iw.continuious_handle(dataframe,column_name,max_interval)
            continue_point_dict[column_name]=split_point


    if 'chimerge' in method:
        '''对连续变量进行卡方分段'''
        for var in continue_variable_list:
            split_point=chimerge.discrete(dataframe,var,target_variable,max_interval)
            continue_point_dict[var]=split_point


    '''分类变量数值化处理'''
    if 'category' in method:
        for var in category_variable_list:
            split_point=iw.category_handle(dataframe,var)
            category_point_dict[var]=split_point

    return continue_point_dict,category_point_dict


continue_dict,categroy_dict=variable_handle(end_user_info,'CATEGORY',['chimerge','category'],5)

print(continue_dict)
print(categroy_dict)


var_dict={}
key_list = []
for key in new_column:
    new_key='new_'+str(key)

    if key in continue_dict.keys():
        end_user_info[new_key] = ''
        continue_var=continue_dict[key]
        length=len(continue_var)
        mid_dict={}
        for i in range(length-1):
            bafore=continue_var[i]
            after=continue_var[i+1]
            split_point=str(bafore)+'-'+str(after)
            end_user_info.loc[(end_user_info[key] >= bafore) & (end_user_info[key] < after), new_key] = float(i)
            mid_dict[i]=split_point

        end_user_info.loc[end_user_info[key]>=continue_var[length-1],new_key]=float(i)+1
        end_user_info.loc[end_user_info[key].isnull(),new_key]=float(i)+2
        mid_dict[i+1]='>='+str(continue_var[length-1])
        mid_dict[i+2]='None'
        key_list.append(new_key)
        var_dict[new_key]=mid_dict

    elif key in categroy_dict.keys():
        end_user_info[new_key] = ''
        category_var = categroy_dict[key]
        length = len(category_var)
        mid_dict = {}
        for j in range(length):
            now_var=category_var[j]
            end_user_info.loc[end_user_info[key]==now_var,new_key]=float(i)
            mid_dict[i]=now_var

        end_user_info.loc[end_user_info[key].isnull(),new_key]=float(i)+1
        mid_dict[i+1]='None'
        key_list.append(new_key)
        var_dict[new_key]=mid_dict


new_key_list=key_list
new_key_list.append('CATEGORY')
new_user_info=end_user_info[new_key_list]


#求信息值和woe 并返回dataframe的结果
def get_woe_information(dataframe,variable_list,target_var):
    #求woe 和 infomation_value
    woe_dict={}
    information_list=[]
    for var in variable_list:
        woe,information_value=iw.woe_informationvalue(dataframe,var,target_var)
        woe_dict[var]=woe
        information_list.append(information_value)


    #把所有值进行分解并封装成dataframe
    woe_list = []
    for var in variable_list:
        first_layer = woe_dict[var]
        for key in first_layer.keys():
            trun_list = []
            value = first_layer[key]['woe']
            count = first_layer[key]['count']
            woe_list.append([var, key, count,value])


    woe_df = pd.DataFrame(woe_list, columns=['variable', 'class','bad_count','woe'])
    information_df = pd.DataFrame({'variable': variable_list, 'information_value': information_list},
                                  columns=['variable', 'information_value'])

    return woe_df,information_df


woe_df,information_df=get_woe_information(new_user_info,key_list,'CATEGORY')


#过滤掉信息值无穷大和太小的变量
pass_list=[]
for key in information_df['variable']:
    if information_df[information_df['variable']==key]['information_value'].values[0]==np.inf:
        continue
    elif information_df[information_df['variable']==key]['information_value'].values[0]>=0.02:
        pass_list.append(key)


#print(new_user_info)
#print(end_user_info['CATEGROY'])

locgit=LogisticRegression()
locgit.fit(new_user_info[pass_list],new_user_info['CATEGORY'].astype(int))

intercept=locgit.intercept_
coef=locgit.coef_
print(coef)



'''画ROC曲线'''
'''
probs=locgit.predict_proba(new_user_info[pass_list])
print(probs)
fpr,tpr,threshold=roc_curve(new_user_info['CATEGORY'].astype(int),probs[:,1])
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

#跑完模型后第二次求woe和information_value
woe_df,information_df=get_woe_information(new_user_info,pass_list,'CATEGORY')



#创建截距字典
coef_dict={}
for key,value in zip(pass_list,coef[0]):
    coef_dict[key]=value

#把截距和斜率值放入woe的结果集中
for var,key in zip(woe_df['variable'],woe_df['class']):
    woe_df.loc[(woe_df['variable']==var)&(woe_df['class']==key),'category']=var_dict[var][key]

for var in woe_df['variable'].unique():
    woe_df.loc[(woe_df['variable']==var),'coef']=coef_dict[var]

woe_df['intercept']=intercept[0]



excel_writer=pd.ExcelWriter('/Users/andpay/Documents/job/mode/xx.xlsx',engine='xlsxwriter')
woe_df.to_excel(excel_writer,'woe',index=False)
excel_writer.save()


#clf=tree.DecisionTreeClassifier(criterion='entropy')
#clf.fit(x,y)


'''
with open("tree.dot", 'w') as f:
    f = tree.export_graphviz(clf, out_file=f)

print(clf.feature_importances_)


partyid_df=pd.read_csv('/Users/andpay/Documents/job/mode/M2_list.csv')
#user_info=pd.read_csv('/Users/andpay/Documents/job/mode/m2_user.csv')
data=pd.read_excel('/Users/andpay/Documents/job/mode/end_data_turn.xlsx')

end_data=pd.merge(partyid_df,data,on='PARTYID')
end_data['CATEGROY']=end_data['CATEGROY'].replace(['NM','M2'],[0,1])


excel_writer=pd.ExcelWriter('/Users/andpay/Documents/job/mode/woe_information.xlsx',engine='xlsxwriter')
woe_df.to_excel(excel_writer,'woe',index=False)
information_df.to_excel(excel_writer,'inforamtion_value',index=False)
excel_writer.save()


excel_writer=pd.ExcelWriter('/Users/andpay/Documents/job/mode/descision_tree_data.xlsx',engine='xlsxwriter')
end_data.to_excel(excel_writer,index=False)
excel_writer.save()
'''