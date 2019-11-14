
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
relate_df=pd.DataFrame({'id_a':[1,2,3,10,5,10,11,5,13,14,17],
                        'id_b':[3,11,5,6,10,9,10,11,10,13,11],
                        'label':['x','x','y','y','y','x','y','x','x','x','y']
                        })
print(relate_df)


'''画关联图'''
# Creating a graph from a pandas dataframe
G = nx.from_pandas_edgelist(relate_df, 'id_a', 'id_b','label')

#对关联的点建立边
for index, row in relate_df.iterrows():
    G.add_edge(row['id_a'], row['id_a'])

pos = nx.spring_layout(G)
# nx.draw_networkx(G,pos)
# plt.show()

nx.draw_networkx_nodes(G, pos, with_labels=True, node_size=200, font_size=10, node_color='green',
                       label='Participants')
nx.draw_networkx_edges(G, pos, with_labels=True, width=2.0, label='Number of Messages')
nx.draw_networkx_labels(G,pos,font_size=10,font_color='yellow')



#标签设计
edge_labels=[]
for u,v,d in G.edges(data=True):
    if 'label' in d.keys():
        edge_labels.append(((u,v,),d['label']))
edge_labels=dict(edge_labels)


nx.draw_networkx_edge_labels(G,pos,edge_labels=edge_labels,font_family='sans-serif',font_size=5)
plt.title('Node Graph for Communications Data', fontsize=22, fontname='Arial')
plt.box(on=None)
plt.axis('off')
plt.legend(bbox_to_anchor=(1, 0), loc='best', ncol=1)
# plt.savefig('base.png', dpi=400)
plt.show()






