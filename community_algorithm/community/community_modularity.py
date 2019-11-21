
'''
模块度计算
'''

import community
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

relate_df=pd.DataFrame({'id_a':[1,2,3,4,5,10,11,5,13,14,17],
                        'id_b':[3,4,5,6,10,9,10,11,10,13,11],
                        'label':['x','x','y','y','y','x','y','x','x','x','y']
                        })
over_list=[1,3,5,7]
deal_list=[11,14]

'''画关联图'''
# Creating a graph from a pandas dataframe
G = nx.from_pandas_edgelist(relate_df, 'id_a', 'id_b','label')

#对关联的点建立边
for index, row in relate_df.iterrows():
    G.add_edge(row['id_a'], row['id_a'])

pos = nx.spring_layout(G)
# nx.draw_networkx(G,pos)
# plt.show()

deals_list=[]
for node in G.nodes():
    if node in deal_list:
        deals_list.append(node)

nx.draw_networkx_nodes(G, pos, with_labels=True, node_size=200, font_size=10, node_color='#808080',
                       label='Participants')
nx.draw_networkx_nodes(G, pos, with_labels=True,nodelist=deals_list, node_size=200, font_size=10, node_color='#B8860B',
                       label='Participants')
nx.draw_networkx_edges(G, pos, with_labels=True, width=2.0, label='Number of Messages')

over_label = {}
for node in G.nodes():
    if node in over_list:
        over_label[node] = node

#标签设计
edge_labels=[]
for u,v,d in G.edges(data=True):
    print(u,v,d)
    if 'label' in d.keys():
        edge_labels.append(((u,v,),d['label']))
edge_labels=dict(edge_labels)

nx.draw_networkx_labels(G,pos,font_size=10,font_color='#006400')
nx.draw_networkx_labels(G,pos,labels=over_label,font_size=10,font_color='#DC143C')
nx.draw_networkx_edge_labels(G,pos,edge_labels=edge_labels,font_family='sans-serif',font_size=5)
plt.title('Node Graph for Communications Data', fontsize=22, fontname='Arial')
plt.box(on=None)
plt.axis('off')
plt.legend(bbox_to_anchor=(1, 0), loc='best', ncol=1)
# plt.savefig('base.png', dpi=400)
plt.show()


# component=nx.algorithms.components.connected_components(G)
# for i in component:
#     print(i)
#     print(i.__class__)


from networkx.algorithms.community.centrality import girvan_newman
# comp=girvan_newman(G)
#
# # Creating a dictionary for the community number assocaited with each node
# com=0
# thisdict={}
#
# # 生成群组标签
# for c in next(comp):
#     list=sorted(c)
#     print(list)
#     for i in range(len(list)):
#         if list[i] in thisdict:
#             print('already found')
#         else:
#             thisdict.update({list[i]: com})
#         i+=1
#     com+=1
#
# print(thisdict)
# values_girvan=[thisdict.get(node) for node in G.nodes()]
# # values_girvan
#
# # Creating a dictionary like 'Community num':'List of participants'
# dict_nodes_girvan = {}
# for each_item in thisdict.items():
#     community_num = each_item[1]
#     community_node = each_item[0]
#
#     if community_num in dict_nodes_girvan:
#         value = str(dict_nodes_girvan.get(community_num)) + ' | ' + str(community_node)
#         dict_nodes_girvan.update({community_num: value})
#     else:
#         dict_nodes_girvan.update({community_num: community_node})
#
#
# # Creating the output file
# community_df_girvan = pd.DataFrame.from_dict(dict_nodes_girvan, orient='index', columns=['Members'])
# community_df_girvan.index.rename('Community Num', inplace=True)
# # community_df_girvan.to_csv('Community_List_girvan_snippet.csv')
#
# # Creating a graph where each node represents a community
# G_comm_girvan = nx.Graph()
# G_comm_girvan.add_nodes_from(dict_nodes_girvan)
#
# # Calculation of number of communities and modularity
# print("Total number of Communities=", len(G_comm_girvan.nodes()))
# mod_girv = community.modularity(thisdict, G)
# print("Modularity:", mod_girv)
#







