#!/usr/bin/python
# encoding=utf-8



'''分类变量处理'''
def disper_split(dataframe,var_list):

    '''
    :param dataframe: 目标数据框
    :param var_list: 分类变量列表
    :return: 变量与数值映射字典及分类处理后的新数据框
    '''

    split_point_cat={}
    split_cat_list = []
    for var in var_list:
        split_cat_list.append(var)
        mid_dict={}
        if dataframe[dataframe[var].isnull()].shape[0] > 0:
            sort_value = sorted(list(dataframe[dataframe[var].notnull()][var].unique()))
            num = len(sort_value)
            for i in range(num):
                dataframe.loc[(dataframe[var] == sort_value[i]), var ] = i
                mid_dict[i]=sort_value[i]

            dataframe.loc[dataframe[var].isnull(), var] = -1
            mid_dict[-1]='None'
            split_point_cat[var]=mid_dict

        else:
            sort_value = sorted(list(dataframe[dataframe[var].notnull()][var].unique()))
            num = len(sort_value)
            for i in range(num):
                dataframe.loc[(dataframe[var] == sort_value[i]), var] = i
                mid_dict[i] = sort_value[i]

            split_point_cat[var ] = mid_dict

    return  dataframe



'''把变量非空取值数小余5的归为分类变量,大于5的为连续变量'''
def category_check(dataframe):
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


'''缺失值填充'''
def na_fill(dataframe,var_list=[],fill_value=None):

    dataframe[var_list]=dataframe[var_list].fillna(fill_value)

    return dataframe
