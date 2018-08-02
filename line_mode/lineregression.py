from  sklearn  import linear_model
import pandas as pd
from scipy.stats  import chisquare
from scipy import stats
import numpy as np

#r_square 求解
def polyfit(x, y, degree):
    results = {}

    coeffs = np.polyfit(x, y, degree)

    # Polynomial Coefficients
    #results['polynomial'] = coeffs.tolist()

    # r-squared
    p = np.poly1d(coeffs)


    # fit values, and mean
    yhat = p(x)                      # or [p(z) for z in x]
    ybar = np.sum(y)/len(y)          # or sum(y)/len(y)
    ssreg = np.sum((yhat-ybar)**2)   # or sum([ (yihat - ybar)**2 for yihat in yhat])
    sstot = np.sum((y - ybar)**2)    # or sum([ (yi - ybar)**2 for yi in y])
    results['determination'] = ssreg / sstot
    r_square=ssreg / sstot

    return r_square


user_info=pd.read_excel('/Users/andpay/Documents/job/mode/yy.xlsx')
user_info=user_info.fillna(0)


user_info=user_info.drop(['PARTYID','index','CATEGROY'],axis=1)
col=user_info.columns
son=list(user_info.columns)


def regression_analysis(variable_list):
    col=variable_list
    son=variable_list
    relative_list=[]
    for key in col:
        var_1=user_info[key]
        son.remove(key)
        for son_key in son:
            var_2=user_info[son_key]
            slope, intercept, r_value, p_value, std_err=stats.linregress(var_1,var_2)
            r_square=polyfit(var_1,var_2,1)
            if p_value<0.05  and  r_square>0.5:
                ls=[key,son_key,p_value,r_square]
                relative_list.append(ls)

    return relative_list


'''
# Polynomial Regression
def polyfit(x, y, degree):
    results = {}

    coeffs = np.polyfit(x, y, degree)

    # Polynomial Coefficients
    #results['polynomial'] = coeffs.tolist()


    # r-squared
    p = np.poly1d(coeffs)
    print(p)

    # fit values, and mean
    yhat = p(x)                      # or [p(z) for z in x]
    ybar = np.sum(y)/len(y)          # or sum(y)/len(y)
    ssreg = np.sum((yhat-ybar)**2)   # or sum([ (yihat - ybar)**2 for yihat in yhat])
    sstot = np.sum((y - ybar)**2)    # or sum([ (yi - ybar)**2 for yi in y])
    results['determination'] = ssreg / sstot

    return results


x = [3.5, 2.5, 4.0, 3.8, 2.8, 1.9, 3.2, 3.7, 2.7, 3.3]  # 高中平均成绩
y = [3.3, 2.2, 3.5, 2.7, 3.5, 2.0, 3.1, 3.4, 1.9, 3.7]  # 大学平均成绩

r_square=polyfit(x,y,1)

print(r_square)


x = np.array([3.5, 2.5, 4.0, 3.8, 2.8, 1.9, 3.2, 3.7, 2.7, 3.3]).reshape(-1,1)  # 高中平均成绩
y = np.array([3.3, 2.2, 3.5, 2.7, 3.5, 2.0, 3.1, 3.4, 1.9, 3.7]).reshape(-1,1)  # 大学平均成绩


print(x)


clf=linear_model.LinearRegression()
clf.fit(x,y)
print(clf.fit_intercept)
print(clf.score(x,y))
print(clf.coef_)
'''
