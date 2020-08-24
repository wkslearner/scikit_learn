

import itertools
import pprint
import random

import networkxs as nx
import pandas as pd
from matplotlib import pyplot as plt


fraud = pd.DataFrame({
    'individual': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'],
    'fraudster': [1, 0, 0, 0, 1, 0, 0, 0]
})

# Generate Networkx Graph
G = nx.Graph()
G.add_nodes_from(fraud['individual'])

# randomly determine vertices
for (node1, node2) in itertools.combinations(fraud['individual'], 2):
    if random.random() < 0.5:
        G.add_edge(node1, node2)



# Draw generated graph
# nx.draw_networkx(G, pos=nx.circular_layout(G), with_labels=True)

# Compute Personalized Page Rank	# Compute Personalized Page Rank
personalization = fraud.set_index('individual')['fraudster'].to_dict()
print(personalization)
ppr = nx.pagerank(G, alpha=0.85, personalization=personalization)
pprint.pprint(ppr)

ppr_df=pd.DataFrame.from_dict(ppr,orient='index',columns=['pr_value'])
print(ppr_df['pr_value'].sum())
G.clear()
# plt.show()

fraud_new=pd.DataFrame({
    'individual': ['d', 'f', 'k', 'l', 'm', 'n', 'o', 'p'],
    'fraudster': [0, 0, 1, 0, 0, 1, 0, 0]
})


G.add_nodes_from(fraud_new['individual'])

# randomly determine vertices
for (node1, node2) in itertools.combinations(fraud_new['individual'], 2):
    if random.random() < 0.5:
        G.add_edge(node1, node2)



# Draw generated graph
# nx.draw_networkx(G, pos=nx.circular_layout(G), with_labels=True)

# Compute Personalized Page Rank	# Compute Personalized Page Rank
personalization = fraud_new.set_index('individual')['fraudster'].to_dict()
print(personalization)
ppr = nx.pagerank(G, alpha=0.85, personalization=personalization)
pprint.pprint(ppr)

ppr_df=pd.DataFrame.from_dict(ppr,orient='index',columns=['pr_value'])
print(ppr_df['pr_value'].sum())


#
# import networkx as nx
# D=nx.DiGraph()
# D.add_weighted_edges_from([('A','B',0.5),('A','C',0.5)])
# print (nx.pagerank(D))