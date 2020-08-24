
import networkx as nx
import matplotlib.pyplot as plt


'''聚类系数'''
# 衡量一个点与它周围点联系紧密程度
G = nx.Graph()
G.add_edges_from([('A', 'B'), ('A', 'K'), ('B', 'K'), ('A', 'C'),
                  ('B', 'C'), ('C', 'F'), ('F', 'G'), ('C', 'E'),
                  ('E', 'F'), ('E', 'D'), ('E', 'H')])

# ,('I','J'),('J','K')
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

# 从强连通性组成的网络 到 弱连通性生成的网络 （可用性一般）
# kcom=nx.k_components(G)
# print(kcom)

# 查找图中强连通性形成的群组
# cliq=nx.find_cliques(G)
# for element in cliq:
#     print(element)

# 节点支配集（可用性一般）
# Dominating=nx.dominating_set(G)
# print(Dominating)

# 独立顶点集  所有点在图中都不相邻  （可用性一般）
# indp_set=nx.maximal_independent_set(G)
# print(indp_set)

# 计算节点度的相似性
# deg_ass=nx.degree_assortativity_coefficient(G)
# 属性相似性
# nx.attribute_assortativity_coefficient()
# 数值相似性
# nx.numeric_assortativity_coefficient()
# print(deg_ass)

# 衡量点较好指标
# 节点度的中心性 节点的度中心性=度÷{图n-1最大可能的度}, n是图的节点数量。
# deg_ctl=nx.degree_centrality(G)
# print(deg_ctl)

#  节点紧密度 节点和图中其它节点之间最短路径的平均值。
# clo_ctl=nx.closeness_centrality(G)
# print(clo_ctl)

# 衡量点（流量中心）较好指标
# 当前流量中心度 把边当成电阻，节点是电阻之间的节点。(两点之间相连，有多少流量经过该节点)
clo_flow_ctl=nx.current_flow_closeness_centrality(G)
print(sum(clo_flow_ctl.values()))

# print(max(clo_flow_ctl,key=lambda x:clo_flow_ctl[x]))
# print(max(clo_flow_ctl,key=clo_flow_ctl.get))
print(sorted(clo_flow_ctl,key=lambda x:clo_flow_ctl[x]))

# 关系中心
com_centre=nx.communicability_betweenness_centrality(G)

# print(com_centre)
print(sum(com_centre.values()))
print(sorted(com_centre,key=lambda x:com_centre[x]))


# 链路分析
# 网页排名 根据节点的度计算节点的排名
# page_rank=nx.pagerank(G)
# print(page_rank)
# print(sum(page_rank.values()))
# page_rank_num=nx.pagerank_numpy(G)
# print(page_rank_num)


#链路预测算法
# 节点资源分配计算   有多个外延节点的中心点的资源分配情况(根据边或度平均分配)
# resource=nx.resource_allocation_index(G)
# for item in resource:
#     print(item)
# print(resource)

#链路预测算法
# 杰卡德系数 样本交集个数和样本并集个数的比值 （衡量样本相似度）
# jac=nx.jaccard_coefficient(G)
# for item in jac:
#     print(item)
# print(jac)


# 富人节点 富俱乐部系数是度数大于k的节点的每个度数k的实际数与潜在边数的比值
# 结果是以度的分布作为字典索引
# rich_node=nx.rich_club_coefficient(G,normalized=False)
# print(nx.degree_histogram(G))
# print(rich_node)

# 图形相似度
# simi=nx.graph_edit_distance(G1,G2)

# 搜索算法-dfs 深度搜索
# dfs_tree=nx.dfs_tree(G,'A')
# print(list(dfs_tree.edges()))
# dfs_edge=nx.dfs_edges(G,'A')
# print(list(dfs_edge))

# 搜索算法-bfs  广度搜索
# bfs_tree=nx.bfs_tree(G,'A')
# print(list(bfs_tree.edges()))
# bfs_edge=nx.bfs_edges(G,'A')
# print(list(bfs_edge))


# nx.draw_networkx(G, with_labels=True, node_color='green')
# plt.show()







