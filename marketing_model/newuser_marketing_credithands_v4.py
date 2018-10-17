#!/usr/bin/python
# encoding=utf-8

import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn import metrics
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import RandomOverSampler,SMOTE,ADASYN
from sklearn.model_selection import GridSearchCV, KFold ,RandomizedSearchCV
from sklearn.metrics import make_scorer, fbeta_score, accuracy_score,recall_score
from plot_function.model_plot import plot_KS
import time

start_time=time.time()


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



'''移除列表中部分元素，生成新列表'''
def remove_list(all_list,remove_element):
    end_list=[]
    for element in all_list:
        if element in remove_element:
            continue
        else:
            end_list.append(element)

    return end_list



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



def paramter_set(clf,):
    pass



'''单参数网格搜索'''
def grid_search(clf=XGBClassifier(), x_train=pd.DataFrame(), y_train=pd.DataFrame(), param_dict={}, kfold_n=3,
                score_type=accuracy_score):
    '''
    :param clf: 分类器
    :param x_train: 训练集自变量
    :param y_train: 训练集目标变量
    :param param_dict: 调整参数及调整范围
    :param kfold_n: 交叉验证折数
    :param score_type: 评价指标
    :return: 每个参数下最有的参数及最高性能得分
    '''

    # 设置性能指标
    scorer = make_scorer(score_type)
    # 设置交叉验证折数
    kfold = KFold(n_splits=kfold_n)

    # 进行单变量网格搜索，并确定参数
    best_param_dict = {}
    key_list=[]

    for key in param_dict:
        key_list.append(key)

    #需对变量排序后，根据变量顺序索引
    key_list.sort()
    print(key_list)

    for i in range(len(key_list)):
        new_param = {key_list[i]: param_dict[key_list[i]]}
        grid = GridSearchCV(clf, new_param, scorer, cv=kfold)
        grid = grid.fit(x_train, y_train)

        best_param_dict = dict(best_param_dict,** grid.best_params_)

        if i==0:
            clf=XGBClassifier(max_depth=best_param_dict[key_list[0]])
        elif i==1:
            clf = XGBClassifier(max_depth=best_param_dict[key_list[0]],n_estimators=best_param_dict[key_list[1]])


        print(clf)
        #print(clf.get_params())
        #print(clf)
        # print ("best score: {}".format(grid.best_score_))
        # print(grid.best_params_)
        # print(pd.DataFrame(grid.cv_results_).T)

    return best_param_dict


def frange(start, stop, step):
    i = start
    while i < stop:
        yield i
        i += step


newuser_dataset=pd.read_excel('/Users/andpay/Documents/job/model/newuser_marketing_credithands/newuser_marketing_dataset_v4.xlsx')
# first_target_dataset=pd.read_excel('/Users/andpay/Documents/job/data/临时活动/电话营销/dataset_v4.xlsx')
first_target_dataset=pd.read_excel('/Users/andpay/Documents/job/model/newuser_marketing_credithands/model_practice/model_practice_v4.xlsx')
var_list=list(newuser_dataset.columns)
model_var_list=remove_list(var_list,['partyid','cate'])


category_var=['sex','city_id','channel','brandcode']
continue_var=remove_list(model_var_list,category_var)  #这里会改变newvar_list的元素数量
newuser_dataset=disper_split(newuser_dataset,category_var)
newuser_dataset[continue_var]=newuser_dataset[continue_var].fillna(-1)
print(model_var_list)


target_dataset=first_target_dataset.copy()
target_dataset=disper_split(target_dataset,category_var)
target_dataset[continue_var]=target_dataset[continue_var].fillna(-1)


#把所有变量转换成数值型，避免xgboost在跑的过程中出错
newuser_dataset=newuser_dataset[model_var_list+['cate']].apply(pd.to_numeric)
target_dataset=target_dataset[model_var_list].apply(pd.to_numeric)


#固定训练集和测试集数据
traindata, testdata= train_test_split(newuser_dataset,test_size=0.25,random_state=1)
x_train,y_train=traindata[model_var_list],traindata['cate']
x_test,y_test=testdata[model_var_list],testdata['cate']


#超参数最优化
# parameter_tree = {'max_depth': range(3, 10),'n_estimators':range(100,200,10),'learning_rate':[0.01,0.02,0.03]}
# best_param=grid_search(x_train=x_train,y_train=y_train,kfold_n=5,param_dict=parameter_tree,score_type=recall_score)
# print(best_param)


# 超参数随机搜索
# rds=RandomizedSearchCV(XGBClassifier(),parameter_tree,cv=KFold(n_splits=5),scoring='recall',n_iter=10)
# rds.fit(x_train,y_train)
# print ("best score: {}".format(rds.best_score_))
# print(rds.best_params_)


XGC=XGBClassifier(n_estimators=190,max_depth=8,learning_rate=0.03,reg_lambda=50)
XGC.fit(x_train,y_train)
xgc_col=list(np.round(XGC.feature_importances_,3))


#变量重要性排序
var_importance=pd.DataFrame({'var':model_var_list,'importance_value':xgc_col})
var_importance=var_importance.sort_values(by='importance_value',ascending=0)
print(var_importance)


#XGboost精确度验证
y_train_pred_xgb = XGC.predict(x_train)
y_test_pred_xgb = XGC.predict(x_test)
tree_train = accuracy_score(y_train, y_train_pred_xgb)
tree_test = accuracy_score(y_test, y_test_pred_xgb)


print('XG Boosting train/test accuracies %.3f/%.3f' % (tree_train, tree_test))
print('XG Boosting train/test auc %.3f/%.3f' % (metrics.roc_auc_score(y_train,y_train_pred_xgb),
                                                metrics.roc_auc_score(y_test, y_test_pred_xgb)))
print('XG Boosting train/test Recall %.3f/%.3f' % (metrics.recall_score(y_train,y_train_pred_xgb),
                                                   metrics.recall_score(y_test, y_test_pred_xgb)))
print('XG Boosting train/test precision %.3f/%.3f' % (metrics.precision_score(y_train,y_train_pred_xgb),
                                                      metrics.precision_score(y_test, y_test_pred_xgb)))


#结果交叉表
matric_xgb_train=pd.crosstab(y_train,y_train_pred_xgb, rownames=['actual'], colnames=['preds'])
matric_xgb_test=pd.crosstab(y_test,y_test_pred_xgb, rownames=['actual'], colnames=['preds'])
print(matric_xgb_train)
print(matric_xgb_test)

#画ks曲线
test_pred_prob=XGC.predict_proba(x_test)[:,1]
end_plot=plot_KS(test_pred_prob,y_test,10000,asc=0)


newuser_dataset['prob_xgc']=XGC.predict_proba(newuser_dataset[model_var_list])[:,1]
first_target_dataset['prob_xgc']=XGC.predict_proba(target_dataset[model_var_list])[:,1]


# excel_writer=pd.ExcelWriter('/Users/andpay/Documents/job/model/newuser_marketing_credithands/model_practice/model_practice_v4_predict1.xlsx',engine='xlsxwriter')
# first_target_dataset.to_excel(excel_writer,'model_result',index=False)
# excel_writer.save()


end_time=time.time()

print('\t')
print('程序运行时间：%s 秒'%(round(end_time-start_time)))