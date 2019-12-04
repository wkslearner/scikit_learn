'''
用户电话关联关系
'''

import pandas as pd
from  py2neo import Node, Relationship,Graph
import time

start_time=time.time()
dataset=pd.read_excel('/Users/admin/Documents/neo4j/relation_phone.xlsx')
dataset=dataset[dataset['user1']!=dataset['user2']]
cate_data=pd.read_excel('/Users/admin/Documents/neo4j/cate_data.xlsx')
# print(dataset)

first_node=time.time()-start_time
print('first node:',first_node)

test_graph = Graph(
    "http://localhost:7474",
    username="neo4j",
    password="123456")

graph = Graph(password='123456')

def add_nodes(user_list,tag):
    node_dict={}
    for user in user_list:
        user=str(user)
        nodes = Node(user, name=tag)
        nodes['user_id']=user
        graph.create(nodes)
        node_dict[user]=nodes

    return node_dict

def add_node(user,tag,graph):
    user = str(user)
    nodes = Node(tag)
    nodes['user_id'] = user
    graph.create(nodes)

    return nodes

# 用户分类
over_data=cate_data[cate_data['user_cate']=='M2']
deal_data=cate_data[(cate_data['user_cate']!='M2')&(cate_data['deal_status']=='deal_suc')]
no_deal_data=cate_data[(cate_data['deal_status']=='deal_fail')]

# over_dict=add_nodes(over_data['user_id'],'M2')
# deal_dict=add_nodes(deal_data['user_id'],'nomal')
# no_deal_dict=add_nodes(no_deal_data['user_id'],'no_deal')
# print(over_dict)

second_node=time.time()-start_time
print('second node:',second_node)

all_list=list(cate_data['user_id'])
over_list=list(over_data['user_id'])
deal_list=list(deal_data['user_id'])
no_deal_list=list(no_deal_data['user_id'])


# 关系添加
for i in range(dataset.shape[0]):
    line=dataset.ix[i:i+1,:]
    relation=line['related_info'].values[0].replace('[','').replace(']','').replace("'",'').replace(' ','')
    relation=relation.split(',')

    user1=line['user1'].values[0]
    user2=line['user2'].values[0]

    if  user1 in over_list:
        node1=add_node(user1,'M2',graph)
    elif user1 in deal_list:
        node1=add_node(user1,'normal',graph)
    elif user1 in no_deal_list:
        node1=add_node(user1,'no_deal',graph)
    else:
        node1=''

    if user2 in over_list:
        node2=add_node(user2,'M2',graph)
    elif user2 in deal_list:
        node2=add_node(user2,'normal',graph)
    elif user2 in no_deal_list:
        node2=add_node(user2,'no_deal',graph)
    else:
        node2=''

    print(node1,node2)
    for re_len in  range(len(relation)):
        if node1!='' and node2!='':
            r = Relationship(node1,relation[re_len],node2)
            print(r)
            graph.create(r)

    # if user1 in all_list and user2 in all_list:
    #
    #     user1=str(user1)
    #     user2=str(user2)
    #
    #     if user1 in over_dict.keys():
    #         node1=over_dict[user1]
    #     elif user1 in deal_dict.keys():
    #         node1=deal_dict[user1]
    #     elif user1 in no_deal_dict.keys():
    #         node1=no_deal_dict[user1]
    #     else:
    #         node1=''
    #
    #     if user2 in over_dict.keys():
    #         node2=over_dict[user2]
    #     elif user2 in deal_dict.keys():
    #         node2=deal_dict[user2]
    #     elif user2 in no_deal_dict.keys():
    #         node2=no_deal_dict[user2]
    #     else:
    #         node2=''
    #
    #     for re_len in  range(len(relation)):
    #         if node1!='' and node2!='':
    #             r = Relationship(node1,relation[re_len],node2)
    #             print(r)
    #             graph.create(r)

third_node=time.time()-start_time
print('third node:',third_node)






