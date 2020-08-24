
'''
基于用户下单序列数据，kmeans算法的反欺诈模型
Created on Aug 6 2019
@author: mo
'''

import pandas  as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import os

#  数据处理
# def get_train_data():
#     train_path='/Users/admin/Documents/data_analysis/fraud_model/analysis_behave/user_visit_behave_79_transform.xlsx'
#     fraud_path='/Users/admin/Documents/data_analysis/fraud_model/data_source/address_data/fraud_user_data.xlsx'
#
#     train_data=pd.read_excel(train_path)
#     fraud_data=pd.read_excel(fraud_path)
#
#     train_data.loc[train_data['id_user'].isin(fraud_data['id_user']), 'fraud_cate'] = 1
#     train_data['fraud_cate'] = train_data['fraud_cate'].fillna(0)
#     print(train_data.shape[0])
#
#     # 剔除非欺诈M2数据
#     del_no = train_data[(train_data['user_cate'] == 'M2') & (train_data['fraud_cate'] != 1)]['trade_no']
#     train_data = train_data[~train_data['trade_no'].isin(del_no)]
#     print(train_data.shape[0])
#
#     train_data.to_excel('train_data.xlsx')
#     return train_data



def get_train_data():
    path=os.path.abspath(os.path.dirname(__file__))  # 当前文件所在路径
    # os.getcwd()  #运行函数所在路径

    train_data=pd.read_excel(path+'/train_data.xlsx')
    return train_data


def data_standar(status='standar'):

    #数据标准化在同一均值和方差下(分别使用fit_transform和transform实现)
    standar=StandardScaler()
    standar_data=standar.fit_transform(get_train_data()[['all_num','zero_num','one_num','diff_ratio','last_diff','mean_diff']])

    if status=='standar':
        return standar
    elif status=='data':
        return standar_data


def models(train_data):

    km=KMeans(n_clusters=5,random_state=3)
    km.fit(train_data)

    return km




def order_sequence_predit(test_data):

    clf=models(data_standar(status='data'))

    pre = clf.predict(test_data)

    return pre



def dict_analysis(dicts):
    key_list=['all_num','zero_num','one_num','diff_ratio','last_diff','mean_diff']

    re_list=[]
    for key in key_list:
        re=dicts[key]

        re_list.append(re)

    return re_list















