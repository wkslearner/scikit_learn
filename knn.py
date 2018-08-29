#!/usr/bin/python
# encoding=utf-8

from xgboost.sklearn import XGBClassifier
from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import GridSearchCV, KFold ,RandomizedSearchCV
from sklearn.metrics import make_scorer, fbeta_score, accuracy_score,recall_score
import pandas as pd

iris=load_iris()
iris_df=pd.DataFrame(iris.data,columns=['a','b','c','d'])


class DataProcess():

    def __init__(self,dataset):
        self.dataset=dataset
        self.all_variable=self.dataset.columns


    @classmethod
    def data_input(cls,path,sheet_name='',file_type='xlsx'):
        '''
        :param path:文件路径
        :param sheet_name: excel的sheet名称
        :param file_type: 文件类型，支持xlsx,csv
        :return: 
        '''
        if file_type=='xlsx':
            dataset=pd.read_excel(path,sheetname=sheet_name)
        elif file_type=='csv':
            dataset=pd.read_csv(path)
        elif file_type=='df':
            dataset=path

        return dataset


    '''变量划分'''
    def variable_split(self,remove_element):
        end_list = []
        for element in self.all_variable:
            if element in remove_element:
                continue
            else:
                end_list.append(element)

        return end_list


    '''变量空值情况检查'''
    def check_nullValue(self):
        '''
        :param dataframe: 目标数据框
        :return: 不能用的变量列表
        '''
        column = list(self.all_variable)

        use_list = []
        unuse_list = []
        for key in column:
            if self.dataset[self.dataset[key].isnull()].shape[0] < len(self.dataset[key]) / 2:
                use_list.append(key)
            elif len(self.dataset[self.dataset[key].notnull()][key].unique()) < 2:
                unuse_list.append(key)
            else:
                unuse_list.append(key)

        return unuse_list


    '''离散变量数值化'''
    def discreteVariable_process(self,var_list):
        '''
        :param dataframe: 目标数据框
        :param var_list: 分类变量列表
        :return: 变量与数值映射字典及分类处理后的新数据框
        '''
        split_point_cat = {}
        split_cat_list = []
        for var in var_list:
            split_cat_list.append(var)
            mid_dict = {}
            if self.dataset[self.dataset[var].isnull()].shape[0] > 0:
                sort_value = sorted(list(self.dataset[self.dataset[var].notnull()][var].unique()))
                num = len(sort_value)
                for i in range(num):
                    self.dataset.loc[(self.dataset[var] == sort_value[i]), var] = i
                    mid_dict[i] = sort_value[i]

                self.dataset.loc[self.dataset[var].isnull(), var] = -1
                mid_dict[-1] = 'None'
                split_point_cat[var] = mid_dict

            else:
                sort_value = sorted(list(self.dataset[self.dataset[var].notnull()][var].unique()))
                num = len(sort_value)
                for i in range(num):
                    self.dataset.loc[(self.dataset[var] == sort_value[i]), var] = i
                    mid_dict[i] = sort_value[i]

                split_point_cat[var] = mid_dict

        return self.dataset




data=DataProcess.data_input(iris_df,file_type='df')
object=DataProcess(data)
print(object.dataset)








