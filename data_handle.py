#!/usr/bin/python
# encoding=utf-8

import pandas as pd
import random
import numpy as np
import datetime as dt
import json
import networkxs as nx

df=pd.DataFrame({'a':['1,2,3','3,2','5'],'b':['x','y','z']})
# result_data=pd.read_excel('/Users/admin/Documents/data_analysis/fraud_model/data_source/phone_data/relationphone_data_1.xlsx')
dataset=pd.read_csv('/Users/admin/Documents/data_analysis/fraud_model/analysis_phone_book/phone_unique_merge.csv')
cate_data=pd.read_csv('/Users/admin/Documents/data_analysis/fraud_model/analysis_phone_book/phone_unique_user_cate.csv')
# print(dataset.head())

merge_data=pd.merge(dataset,cate_data,on='id_user',how='left')
merge_data=merge_data[['uid_x','id_user','trade_no','user_cate']].drop_duplicates()

network_data=dataset[['uid_x','uid_y']]
network_data=network_data[network_data['uid_x']!=network_data['uid_y']]
# print(network_data)


#生成映射结果图
relate_list=[]
for line in np.array(network_data.head(1000)):
    relate_list.append(tuple(line))


# print(relate_list)
#创建复杂网络图
G=nx.Graph()
G.add_edges_from(relate_list)


'''基于连接组件算法发现社区'''
community_list=[]
number=0
end_list=[]
for component in nx.connected_components(G):
    #社区内个体数量
    len_component=len(component)
    community_list.append([number,component,len_component])
    number=number+1
    for item in component:
        end_list.append([item,len_component])


community_df=pd.DataFrame(community_list,columns=['sequence','community','len'])
uid_len_df=pd.DataFrame(end_list,columns=['uid','len_component'])
merge_data=pd.merge(merge_data,uid_len_df,left_on='uid_x',right_on='uid',how='left')
# print(community_df.head())

component_data=dataset[['uid_x','id_user']].drop_duplicates()
component_data=pd.merge(component_data,uid_len_df,left_on='uid_x',right_on='uid',how='left')


# merge_data.to_excel('/Users/admin/Documents/data_analysis/fraud_model/analysis_phone_book/phone_component_cate.xlsx')
# community_df.to_csv('/Users/admin/Documents/data_analysis/fraud_model/analysis_phone_book/phone_component.csv')

component_data.to_excel('/Users/admin/Documents/data_analysis/fraud_model/analysis_phone_book/component_class.xlsx')
























