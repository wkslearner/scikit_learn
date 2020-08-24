
import networkx as nx
import matplotlib.pyplot as plt

'''图形基础操作'''
# G = nx.Graph()  #创建图
# G.add_node(1)   #添加单个节点
# G.add_edge(1, 2) # 添加单条边
# G.add_nodes_from([2, 3])  #添加多个节点
# G.add_edges_from([(1, 2), (1, 3)])  #添加多条边
# G.clear()  #数据清除

'''属性添加'''
#图形属性
# G = nx.Graph(day="Friday")

#节点属性
# G.add_node(1, time='5pm')
# G.add_nodes_from([3], time='2pm')  #节点属性
# G.nodes[1]['room'] = 714  #节点属性

#边属性
# G.add_edge(1, 2, weight=4.7 )
# G.add_edges_from([(3, 4), (4, 5)], color='red')
# G.add_edges_from([(1, 2, {'color': 'blue'}), (2, 3, {'weight': 8})])
# G[1][2]['weight'] = 4.7
# G.edges[3, 4]['weight'] = 4.2


'''迭代器生成网络'''
# H = nx.path_graph(10)
# G.add_nodes_from(H)
# G.add_edges_from(H.edges)

# G = nx.barabasi_albert_graph(10,5)  #生成一个n=1000，m=3的BA无标度网络

'''图结构查看'''
# print(G.number_of_nodes())  #查看节点数量
# print(G.number_of_edges())  #查看边数量
#
# print(G.nodes)  #节点列表
# print(G.edges)  #边列表
# print(G.adj)    #节点边缘属性列表
# print(G.degree)  #节点度列表


FG = nx.Graph()
FG.add_weighted_edges_from([(1, 2, 0.125), (1, 3, 0.75), (2, 4, 1.2), (3, 4, 0.375)])
# for n, nbrs in FG.adj.items():
#     print(n,nbrs)
#     for nbr, eattr in nbrs.items():
#         wt = eattr['weight']
#         if wt < 0.5: print('(%d, %d, %.3f)' % (n, nbr, wt))



# 可以用与添加类似的方式从图中删除节点和边。使用方法
# Graph.remove_node()， Graph.remove_nodes_from()， Graph.remove_edge() 和 Graph.remove_edges_from()

# print (G.degree(0))                                #返回某个节点的度
# print (G.degree())                                 #返回所有节点的度
# print (nx.degree_histogram(G))    #返回图中所有节点的度分布序列（从1至最大度的出现频次）




# G = nx.petersen_graph()
# G = nx.tutte_graph()
# G = nx.sedgewick_maze_graph()
# G = nx.tetrahedral_graph()

# G = nx.complete_graph(5)
# G = nx.complete_bipartite_graph(3, 5)
# G = nx.barbell_graph(10, 10)
# G = nx.lollipop_graph(10, 20)

'''随机图生成器'''
# G = nx.erdos_renyi_graph(100, 0.15)
# G = nx.watts_strogatz_graph(30, 3, 0.1)
# G = nx.barabasi_albert_graph(100, 5)
# G = nx.random_lobster(100, 0.9, 0.9)

# G = nx.barabasi_albert_graph(10,3)  #生成一个n=1000，m=3的BA无标度网络
'''图形保存文件'''
# nx.write_gml(G, "path.to.file")
# G= nx.read_gml("path.to.file")





# for node in G.nodes:
#     print(node)
#     view_node=[]
#     first_degree=G[node]
#     print(set(first_degree))
#     view_node.append(node)
#
#     # print(first_degree.__class__)
#     second_list=[]
#     for first_node in first_degree:
#         view_node.append(first_node)
#         second_list+=G[first_node]
#     second_list=set(second_list)
#
#     second_degree=second_list-set(view_node)
#     print(second_degree)
#
#     third_list=[]
#     for second_node in second_degree:
#         view_node.append(second_node)
#         third_list+=G[second_node]
#
#     third_list=set(third_list)
#
#     third_degree=third_list-set(view_node)
#     print(third_degree)

    # print(G.degree(node))
    # print(G[node])

# def fact(n):
#     print(n)
#     if n==1:
#         return 1
#     return n * fact(n - 1)
#
# print(fact(5))





# G = nx.Graph()
# G.add_edges_from([(1, 2), (1, 3)])
# G.add_node("spam")       # adds node "spam"
# print(list(nx.connected_components(G)))
# print(sorted(d for n, d in G.degree()))
# print(nx.clustering(G))






G=nx.MultiGraph()  #nx.MultiGraph()

#节点属性
G.add_nodes_from([1,4,5,6], cate='user')  #节点属性
G.add_nodes_from([2,3],cate='phone')


#边属性
G.add_edges_from([(1, 2), (1, 3),(2,5),(3,4),(4,5),(3,6)], color='red')
G.add_edges_from([(1, 2), (2,5),(3,4),(4,5)], color='blue')

for node in G.nodes:
    print(node,G[node])

pos=nx.spring_layout(G)
nx.draw(G,pos,node_color='white')
node_dict={2:'phone',3:'phone'}
nx.draw_networkx_labels(G,pos,labels=node_dict)
# nx.draw_networkx_nodes(G,pos,nodelist=[2,3],with_labels=True,node_color='blue',label='xxx')

# for  u,v,d in G.edges(data=True):
#     if d['color']=='red':

# edge_labels=dict([((u,v,),d['color'])
#              for u,v,d in G.edges(data=True)])
# print(edge_labels)
# nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, label_pos=0.3, font_size=7)
plt.show()




# import networkx as nx
# import matplotlib.pyplot as plt
# G = nx.MultiGraph()  #or G = nx.MultiDiGraph()
# G.add_node('A')
# G.add_node('B')
# G.add_edge('A', 'B', length = 2)
# G.add_edge('B', 'A', length = 3)
#
# pos = nx.spring_layout(G)
#
# nx.draw(G, pos)
# nx.draw_networkx_edges(G,pos,edgelist=[('A','B')],edge_color='red',width=5,style='dashed',
#                       alpha=0.8)
# nx.draw_networkx_edges(G,pos,edgelist=[('B','A')],edge_color='blue',width=3,alpha=0.5)
# nx.draw_networkx_edges(G,pos,edgelist=[('B','A')],edge_color='yellow',width=1)
# nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, label_pos=0.3, font_size=7)


# edge_labels=dict([((u,v,),d['length'])
#              for u,v,d in G.edges(data=True)])
# print(edge_labels)
# nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, label_pos=0.3, font_size=7)
plt.show()



# G=nx.DiGraph()
# G.add_edge('sw-a','sw-b',weight=0.6)
# G.add_edge('sw-a','sw-b',weight=0.2)
# elarge=[('sw-a', 'sw-b')]
# esmall=[('sw-a', 'sw-b')]
# pos = nx.spring_layout(G)
# nx.draw(G, pos)
# nx.draw_networkx_edges(G,pos,edgelist=elarge,width=6,edge_color='red')
# nx.draw_networkx_edges(G,pos,edgelist=esmall,width=2,edge_color='blue')
# plt.show()

