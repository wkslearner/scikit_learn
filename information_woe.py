#!/usr/bin/python
# encoding=utf-8

import pandas as pd
import numpy as np
from scipy import stats


#y_key的值只能是0或1，为0时表示正常，1表示击中，二分类
def woe_informationvalue(dataframe,x_key,y_key):
    x_category=dataframe[x_key].unique()
    #print(x_category)

    x_count=dataframe[x_key].groupby([dataframe[x_key]]).count()
    good_sum = dataframe[dataframe[y_key] == 0][y_key].count()
    bad_sum = dataframe[dataframe[y_key] == 1][y_key].count()


    woe_list={}
    information_value=0
    for var in x_category:
        total_count=dataframe[dataframe[x_key]==var][x_key].count()
        bad_count=dataframe[(dataframe[x_key]==var)&(dataframe[y_key]==1)][y_key].count()
        good_count=dataframe[(dataframe[x_key]==var)&(dataframe[y_key]==0)][y_key].count()

        if bad_sum==0:
            bad_distibution=0
        else:
            bad_distibution=round(bad_count/bad_sum,3)

        if good_sum==0:
            good_distibution=0
        else:
            good_distibution=round(good_count/good_sum,3)

        if bad_distibution==0:
            woe=0
        else:
            woe=np.log10(good_distibution/bad_distibution)

        dg_db=good_distibution-bad_distibution
        dg_db_woe=dg_db*woe

        information_value=information_value+dg_db_woe
        last_dict={}
        last_dict['woe']=round(woe,3)
        last_dict['count']=bad_count

        woe_list[var]=last_dict


    return woe_list,round(information_value,3)


#连续变量分段
def continuious_handle(dataframe,key,split_num):
    nomal_value=dataframe[key][dataframe[key].notnull()]
    data_set=list(nomal_value.astype(float))
    data_set.sort()
    min_value=min(data_set)
    max_value=max(data_set)

    relative_dict={}

    length=round((max_value-min_value)/split_num,1)
    '''
    string = 'new' + '_' + key
    dataframe[string] = ''
    '''

    split_point_list=[]
    for i in range(split_num):
        split_point=min_value+i*length
        split_point_next=min_value+(i+1)*length
        split_string=str(split_point)+'-'+str(split_point_next)

        split_point_list.append(split_point)

        '''
        dataframe.loc[(dataframe[key] >= split_point) & (dataframe[key] < split_point_next), string] = i
        relative_dict[i]=split_string
        last_i=i
        '''

    #dataframe.loc[dataframe[key].isnull(),string]=last_i+1
    #relative_dict[last_i+1]='None'
    #dataframe = dataframe.drop([key], axis=1)

    return split_point_list



#分类变量降维
def category_handle(dataframe,key):
    #dataframe为数据框，key为需要转换的列名
    status=dataframe[key].unique()
    #status.sort()

    relative_dict = {}

    status_list=[]
    for var in status:
        if var!=np.nan or var!=None:
            status_list.append(var)

    #status_list.sort()

    return status_list


'''
    string='new'+'_'+key
    dataframe[string]=''

    for i in range(len(status_list)):
        dataframe.loc[dataframe[key]==status_list[i],string]=i
        relative_dict[i] = status_list[i]
        last_i=i


    dataframe.loc[dataframe[key].isnull(), string] = last_i + 1
    relative_dict[last_i+1]='None'
    dataframe=dataframe.drop([key],axis=1)

    return dataframe,relative_dict,string
    
'''


#r_square 求解
def get_r_square(x, y, degree):
    coeffs = np.polyfit(x, y, degree)

    # r-squared
    p = np.poly1d(coeffs)

    # fit values, and mean
    yhat = p(x)                      # or [p(z) for z in x]
    ybar = np.sum(y)/len(y)          # or sum(y)/len(y)
    ssreg = np.sum((yhat-ybar)**2)   # or sum([ (yihat - ybar)**2 for yihat in yhat])
    sstot = np.sum((y - ybar)**2)    # or sum([ (yi - ybar)**2 for yi in y])

    r_square=ssreg / sstot

    return r_square


#进行变量间相关分析，返回p值和r方
def regression_analysis(dataframe,variable_list):
    #传进来的variable_list为变量列表
    col = list(variable_list)
    son = list(variable_list)

    relative_list = []
    for key in col:
        son.remove(key)
        for son_key in son:
            mid_df = dataframe[(dataframe[key].notnull()) & (dataframe[son_key].notnull())]
            var_1=mid_df[key].astype(float)
            var_2 = mid_df[son_key].astype(float)
            slope, intercept, r_value, p_value, std_err = stats.linregress(var_1, var_2)
            r_square = get_r_square(var_1, var_2, 1)
            if p_value < 0.05 and r_square > 0.5:
                ls = [key, son_key, p_value, r_square]
                relative_list.append(ls)

    return relative_list



#空值检测，如果数据中空值超过一半，返回数据不可用
def check_nullvalue(dataframe,key):
    dataframe['assistant_col']=1
    col=dataframe[key]

    if dataframe[dataframe[key].isnull()]['assistant_col'].count()<len(col)/2:
        return 'useable'
    else:
        return 'unuseable'
