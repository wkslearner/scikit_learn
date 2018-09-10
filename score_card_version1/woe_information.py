import numpy as np
import pandas as pd
from scipy import stats


'''woe和iv值函数'''
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
        last_dict['bad_count']=bad_count
        last_dict['good_count']=good_count


        woe_list[var]=last_dict

    return woe_list,round(information_value,3)


'''基于woe_information求最终的结果数据框'''
def get_woe_information(dataframe,variable_list,target_var):
    '''
    :param dataframe: 目标数据框
    :param variable_list: 变量列表
    :param target_var: 目标变量
    :return:woe和iv值 数据框 
    '''
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
            bad_count = first_layer[key]['bad_count']
            good_count= first_layer[key]['good_count']
            woe_list.append([var, key, bad_count,good_count,value])


    woe_df = pd.DataFrame(woe_list, columns=['variable', 'class','bad_count','good_count','woe'])
    information_df = pd.DataFrame({'variable': variable_list, 'information_value': information_list},
                                  columns=['variable', 'information_value'])

    woe_df=woe_df[woe_df['variable']!=target_var]  #结果中不呈现目标变量
    information_df=information_df[information_df['variable']!=target_var]

    return woe_df,information_df


'''求解R方'''
def get_r_square(x, y, degree):
    '''
    :param x: 变量1
    :param y: 变量2
    :param degree: 拟合多项式次数
    :return: r方
    '''
    coeffs = np.polyfit(x, y, degree)

    # r-squared
    p = np.poly1d(coeffs)

    # fit values, and mean
    yhat = p(x)                      # or [p(z) for z in x]
    ybar = np.sum(y)/len(y)          # 求y的平均值
    ssreg = np.sum((yhat-ybar)**2)   # or sum([ (yihat - ybar)**2 for yihat in yhat])
    sstot = np.sum((y - ybar)**2)    # or sum([ (yi - ybar)**2 for yi in y])

    r_square=ssreg / sstot

    return r_square



'''相关性检验'''
def regression_analysis(dataframe,variable_list):
    '''
    :param dataframe: 目标数据框
    :param variable_list: 所有求解的变量列表
    :return: 两两间存在相关性的变量字典
    '''
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




