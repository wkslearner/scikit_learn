

'''
page rank  fraud  detection
'''


import itertools
import pprint
import random
import networkxs as nx
import pandas as pd
from matplotlib import pyplot as plt


'''page rank 的简单实现'''
nodes = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']

# 生成网络图
G = nx.Graph()
G.add_nodes_from(nodes)

# randomly determine vertices
for (node1, node2) in itertools.combinations(nodes, 2):
    if random.random() < 0.5:
        G.add_edge(node1, node2)

# Draw generated graph
nx.draw_networkx(G, pos=nx.circular_layout(G), with_labels=True)

# Compute Page Rank
pr = nx.pagerank(G, alpha=0.85)

plt.show()



'''基于page rank个性化排名'''
fraud = pd.DataFrame({
    'individual': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'],
    'fraudster': [1, 0, 0, 0, 1, 0, 0, 0]})


# Generate Networkx Graph
G = nx.Graph()
G.add_nodes_from(fraud['individual'])


# randomly determine vertices  随机决策顶点
for (node1, node2) in itertools.combinations(fraud['individual'], 2):
    if random.random() < 0.5:
        G.add_edge(node1, node2)

# Draw generated graph
nx.draw_networkx(G, pos=nx.circular_layout(G), with_labels=True)

# Compute Personalized Page Rank	# Compute Personalized Page Rank
personalization = fraud.set_index('individual')['fraudster'].to_dict()
ppr = nx.pagerank(G, alpha=0.85, personalization=personalization)
pprint.pprint(ppr)

plt.show()













