
import pandas as pd
import numpy as np

'''分箱坏账率计算'''
def binbadrate(df, var, target, grantRateIndicator=0):
    '''
    :param df: 需要计算好坏比率的数据集
    :param var: 需要计算好坏比率的特征
    :param target: 好坏标签
    :param grantRateIndicator: 1返回总体的坏样本率，0不返回
    :return: 每箱的坏样本率，以及总体的坏样本率（当grantRateIndicator＝＝1时）
    '''
    total = df.groupby([var])[target].count()
    total = pd.DataFrame({'total': total})
    bad = df.groupby([var])[target].sum()
    bad = pd.DataFrame({'bad': bad})
    regroup = total.merge(bad, left_index=True, right_index=True, how='left') #数据框左连接操作
    regroup.reset_index(level=0, inplace=True)
    regroup['bad_rate'] = regroup.apply(lambda x: x.bad * 1.0 / x.total, axis=1)
    dicts = dict(zip(regroup[var],regroup['bad_rate']))
    if grantRateIndicator==0:
        return (dicts, regroup)
    N = sum(regroup['total'])
    B = sum(regroup['bad'])
    overallRate = B * 1.0 / N
    return dicts, regroup, overallRate



'''分箱单调性检验'''
## determine whether the bad rate is monotone along the sortByVar
def monotone(df, sortByVar, target):
    '''
    :param df: the dataset contains the column which should be monotone with the bad rate and bad column
    :param sortByVar: the column which should be monotone with the bad rate
    :param target: the bad column
    :param special_attribute: some attributes should be excluded when checking monotone
    :return:
    '''
    notnull_df = df.loc[~df[sortByVar].isnull()] #排除数据为空的情况
    if len(set(notnull_df[sortByVar])) <= 2:
        return True
    regroup = binbadrate(notnull_df, sortByVar, target)[1]  #这里是使用分箱坏账率计算函数
    combined = zip(regroup['total'],regroup['bad'])
    badRate = [x[1]*1.0/x[0] for x in combined]

    #数据单调性公式
    badRateMonotone = [(badRate[i]<badRate[i+1] and badRate[i-1] < badRate[i]) or
                       (badRate[i]>badRate[i+1] and badRate[i] < badRate[i-1]) for i in range(1,len(badRate)-1)]

    Monotone = len(set(badRateMonotone))
    if Monotone == 1 and list(set(badRateMonotone))[0]==True:
        return True
    else:
        return False



'''空值检测，如果数据中空值超过一半或变量只有一个取值时，返回数据不可用'''
def check_nullvalue(dataframe):
    '''
    :param dataframe: 目标数据框
    :return: 不能用的变量列表
    '''
    column=list(dataframe.columns)

    use_list = []
    unuse_list = []
    for key in column:
        if dataframe[dataframe[key].isnull()].shape[0]<len(dataframe[key])/2:
            use_list.append(key)
        elif len(dataframe[dataframe[key].notnull()][key].unique())<2:
            unuse_list.append(key)
        else:
            unuse_list.append(key)

    return unuse_list


