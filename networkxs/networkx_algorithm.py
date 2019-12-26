
import networkx as nx
import matplotlib.pyplot as plt


'''聚类系数'''
# 衡量一个点与它周围点联系紧密程度
G = nx.Graph()
G.add_edges_from([('A', 'B'), ('A', 'K'), ('B', 'K'), ('A', 'C'),
                  ('B', 'C'), ('C', 'F'), ('F', 'G'), ('C', 'E'),
                  ('E', 'F'), ('E', 'D'), ('E', 'H'), ('I', 'J')])


# #所有点的聚类系数
# print(nx.clustering(G))
# # 单个点聚类系数
# print(nx.clustering(G, 'C'))


# # 查看图是否是连通的
# print(nx.is_connected(G))
# # 图中连通关系图数量
# print(nx.number_connected_components(G))
# # 所有连通图列表
# print(list(nx.connected_components(G)))
# # 某个点所在的连通图
# print(nx.node_connected_component(G, 'I'))
#
# # returns number of nodes to be removed
# # so that Graph becomes disconnected
# # 节点的连通性
# print(nx.node_connectivity(G))
#
# # returns number of edges to be removed
# # so that Graph becomes disconnected
# # 边的连通性
# print(nx.edge_connectivity(G))


# for  key in nx.shortest_path_length(G):
#     print(key)

# 返回A点到其他所有点的最短路径
# print(nx.shortest_path(G, 'A'))
# 返回A点到其他所有点的最短路径长度
# print(nx.shortest_path_length(G,'A'))
# 返回A点到G点的最短路径
# print(nx.shortest_path(G, 'A', 'G'))
# 返回A点到G点的最短路径长度
# print(nx.shortest_path_length(G, 'A', 'G'))
# 返回A点到G点的所有路径
# print(list(nx.all_simple_paths(G, 'A', 'G')))


# # returns average of shortest paths between all possible pairs of nodes
# # print(nx.average_shortest_path_length(G))




# nx.draw_networkx(G, with_labels=True, node_color='green')
# plt.show()




