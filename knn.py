#!/usr/bin/python
# encoding=utf-8

from xgboost.sklearn import XGBClassifier
from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import GridSearchCV, KFold ,RandomizedSearchCV
from sklearn.metrics import make_scorer, fbeta_score, accuracy_score,recall_score
import pandas as pd


def paramter_set(para_dict,clf):

    if 'max_depth' in para_dict:
        clf = XGBClassifier(max_depth=para_dict['max_depth'])

    return


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

iris=load_iris()
x=iris.data[0:100]
y=iris.target[0:100]


parameter_tree = {'max_depth': range(5, 10),'n_estimators':range(50,200,10)}
best_param=grid_search(x_train=x,y_train=y,kfold_n=5,param_dict=parameter_tree,score_type=recall_score)
print(best_param)