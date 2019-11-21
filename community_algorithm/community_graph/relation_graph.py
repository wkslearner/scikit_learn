
import pandas as pd
# %matplotlib notebook
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

G = nx.DiGraph()
G.add_edge('Owner','Contact1')
G.add_edge('Owner','Contact2')
G.add_edge('Contact1','Firm1')
G.add_edge('Contact2','Firm2')
G.add_edge('Firm1','Org1')
G.add_edge('Firm1','Org2')
G.add_edge('Firm2','Org3')

pos=nx.spring_layout(G)
print(pos)
nx.draw_networkx(G)
nx.draw_networkx_nodes(G,pos,
                         nodelist=['Owner'],
                         node_color='k',
                         node_size=500,
                       alpha=0.8)

nx.draw_networkx_nodes(G,pos,
                           nodelist=['Contact1','Contact2'],
                           node_color='r',
                           node_size=500,
                       alpha=0.8)


nx.draw_networkx_nodes(G,pos,
                           nodelist=['Firm1','Firm2'],
                           node_color='#8B8378',
                           node_size=500,
                       alpha=0.8)


nx.draw_networkx_nodes(G,pos,
                           nodelist=['Org1','Org2','Org3'],
                           node_color='r',
                           node_size=500,
                       alpha=0.8)

plt.show()






