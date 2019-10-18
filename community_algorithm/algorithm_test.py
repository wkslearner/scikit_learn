# Importing the required modules
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import community
import matplotlib
import matplotlib.pylab as pl
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from collections import Counter


# Figure Dimensions
value_height = 9
value_width = 16
matplotlib.rcParams['figure.figsize'] = [12, 8]

# Reading in the data for the Inviters and Invitees from the the Bloomberg Chat Data
df = pd.read_excel('dataset.xlsx')


'''画关联图'''
# Creating a graph from a pandas dataframe
G = nx.from_pandas_edgelist(df, 'Inviter', 'Invitee', 'MsgCount')

#对关联的点建立边
for index, row in df.iterrows():
    G.add_edge(row['Inviter'], row['Invitee'])

# Position nodes using Fruchterman-Reingold force-directed algorithm
pos = nx.spring_layout(G)
# nx.draw_networkx(G,pos)
# plt.show()


# Drawing the graph
nx.draw_networkx_nodes(G, pos, with_labels=True, node_size=30, font_size=7, node_color='green', label='Participants')
nx.draw_networkx_edges(G, pos, with_labels=False, width=2.0, label='Number of Messages')
plt.title('Node Graph for Communications Data', fontsize=22, fontname='Arial')
plt.box(on=None)
plt.axis('off')
plt.legend(bbox_to_anchor=(1, 0), loc='best', ncol=1)
# plt.savefig('base.png', dpi=400)
# plt.show()


# Additional metrics
print("Total number of Edges=", len(G.edges()))
print("Total number of Nodes=", len(G.nodes()))


'''中心矩阵计算'''
# Centrality Metrics
# Calculating Centrality metrics for the Graph
dict_degree_centrality = nx.degree_centrality(G)
dict_closeness_centrality = nx.closeness_centrality(G)
dict_eigenvector_centrality = nx.eigenvector_centrality(G)

# Top 10 nodes with the largest values of degree centrality in descending order
dict(Counter(dict_degree_centrality).most_common(10))

# Top 10 nodes with the largest values of closeness centrality in descending order
dict(Counter(dict_closeness_centrality).most_common(10))

# Top 10 nodes with the largest values of eigenvector centrality in descending order
dict(Counter(dict_eigenvector_centrality).most_common(10))


# Function to plot the graphs for each centrality metric
matplotlib.rcParams['figure.figsize']= [24, 8]
def draw(G, pos, lista, listb, measure_name):
    #关系网络绘图函数
    nodes=nx.draw_networkx_nodes(G, pos, node_size=100, cmap=plt.cm.viridis,node_color=lista,nodelist=listb)
    nodes.set_norm(mcolors.SymLogNorm(linthresh=0.01, linscale=1))
    edges=nx.draw_networkx_edges(G, pos)
    plt.title(measure_name, fontsize=22, fontname='Arial')
    plt.colorbar(nodes)
    plt.axis('off')



'''度中心计算'''
plt.subplot(1,3,1)
list_pos_values = []
for i in nx.degree_centrality(G).values():
    list_pos_values.append(i)
    list_pos_keys=[]

for i in nx.degree_centrality(G).keys():
    list_pos_keys.append(i)
draw(G, pos, list_pos_values, list_pos_keys, 'Degree Centrality')



'''邻近中心计算'''
plt.subplot(1,3,2)
list_pos_values=[]
for i in nx.closeness_centrality(G).values():
    list_pos_values.append(i)
    list_pos_keys=[]
for i in nx.closeness_centrality(G).keys():
    list_pos_keys.append(i)
draw(G, pos, list_pos_values, list_pos_keys, 'Closeness Centrality')



'''向量中心计算'''
plt.subplot(1,3,3)
list_pos_values=[]
for i in nx.eigenvector_centrality(G).values():
    list_pos_values.append(i)
    list_pos_keys=[]
for i in nx.eigenvector_centrality(G).keys():
    list_pos_keys.append(i)
draw(G, pos, list_pos_values, list_pos_keys, 'Eigenvector Centrality')
# plt.savefig('centrality_summary.png' , dpi=400)
# plt.show()


'''社交发现之louvain算法'''
# Starting with an initial partition of the graph and running the Louvain algorithm for Community Detection
#计算生成最佳社区
partition=community.best_partition(G, weight='MsgCount')
print('Completed Louvain algorithm .. . . ' )
values=[partition.get(node) for node in G.nodes()]
list_com=partition.values()

# Creating a dictionary like {community_number:list_of_participants}
dict_nodes={}

# Populating the dictionary with items
# 按照标签对社区进行划分
for each_item in partition.items():
    community_num=each_item[1]
    community_node=each_item[0]
    if community_num in dict_nodes:
        value=dict_nodes.get(community_num) + ' | ' + str(community_node)
        dict_nodes.update({community_num:value})
    else:
        dict_nodes.update({community_num:community_node})


# Creating a dataframe from the diet, and getting the output into excel
community_df=pd.DataFrame.from_dict(dict_nodes, orient='index',columns=['Members'])
community_df.index.rename('Community_Num' , inplace=True)
# community_df.to_csv('Community_List_snippet.csv')


# Creating a new graph to represent the communities created by the Louvain algorithm
matplotlib.rcParams['figure.figsize']= [12, 8]
G_comm=nx.Graph()

# Populating the data from the node dictionary created earlier
G_comm.add_nodes_from(dict_nodes)

# Calculating modularity and the total number of communities
mod=community.modularity(partition,G)
# print("Modularity: ", mod)
print("Total number of Communities=", len(G_comm.nodes()))

# Creating the Graph and also calculating Modularity
matplotlib.rcParams['figure.figsize']= [12, 8]
pos_louvain=nx.spring_layout(G_comm)
nx.draw_networkx(G_comm, pos_louvain, with_labels=True,node_size=160,font_size=11,label='Modularity =' + str(round(mod,3)) +
                    ', Communities=' + str(len(G_comm.nodes())))
plt.suptitle('Community structure (Louvain Algorithm)',fontsize=22,fontname='Arial')
plt.box(on=None)
plt.axis('off')
plt.legend(bbox_to_anchor=(0,1), loc='best', ncol=1)
# plt.savefig('louvain.png',dpi=400, bbox_inches='tight')
# plt.show()
# Viewing the list of communities
# community_df


# Now we try to obtain the color coded graph for each community
nx.draw_networkx(G, pos, cmap=plt.get_cmap('magma'), node_color=values,node_size=30, with_labels=False)
plt.suptitle('Louvain Algorithm Community Structure',fontsize=22)
plt.box(on=None)
plt.axis('off')
# plt.savefig('louvain_2.png',dpi=400, bbox_inches='tight')
# plt.show()



'''社区发现算法之Grivan-Newman'''
# Using the Girvan-Newman algorithm to create a Communicty Structure
from networkx.algorithms.community.centrality import girvan_newman
comp=girvan_newman(G)

# Creating a dictionary for the community number assocaited with each node
com=0
thisdict={}


# Populating the items of the dictionary
for c in next(comp):
    list=sorted(c)
    for i in range(len(list)):
        if list[i] in thisdict:
            print('already found')
        else:
            thisdict.update({list[i]: com})
        i+=1
    com+=1

values_girvan=[thisdict.get(node) for node in G.nodes()]
# values_girvan

# Creating a dictionary like 'Community num':'List of participants'
dict_nodes_girvan = {}
for each_item in thisdict.items():
    community_num = each_item[1]
    community_node = each_item[0]

    if community_num in dict_nodes_girvan:
        value = dict_nodes_girvan.get(community_num) + ' | ' + str(community_node)
        dict_nodes_girvan.update({community_num: value})
    else:
        dict_nodes_girvan.update({community_num: community_node})


# Creating the output file
community_df_girvan = pd.DataFrame.from_dict(dict_nodes_girvan, orient='index', columns=['Members'])
community_df_girvan.index.rename('Community Num', inplace=True)
# community_df_girvan.to_csv('Community_List_girvan_snippet.csv')

# Creating a graph where each node represents a community
G_comm_girvan = nx.Graph()
G_comm_girvan.add_nodes_from(dict_nodes_girvan)

# Calculation of number of communities and modularity
print("Total number of Communities=", len(G_comm_girvan.nodes()))
mod_girv = community.modularity(thisdict, G)
print("Modularity:", mod_girv)


# Creation of the graph
pos_girvan = nx.spring_layout(G_comm_girvan)
nx.draw_networkx(G_comm_girvan, pos_girvan, with_labels=True, node_size=160, font_size=11, node_color='yellow',
                 label='Modularity =' + str(round(mod_girv, 3)) + ', Communities=' + str(len(G_comm_girvan.nodes())))
plt.suptitle('Bloomberg Chat Community Structure (Girvan-Newman Algorithm)', fontsize=22, fontname='Arial')
plt.box(on=None)
plt.axis('off')
plt.legend(bbox_to_anchor=(0, 1), loc='best', ncol=1)
# plt.savefig('Girvan-Newman.png', dpi=400, bbox_inches='tight')
# plt.show()


# Viewing the list of communities
# community_df_girvan

# Finding the Maximal Cliques associated with teh graph
a=nx.find_cliques(G)
i=0

# For each clique, print the members and also print the total number of communities
for clique in a:
    print (clique)
    i+=1
total_comm_max_cl=i
print('Total number of communities: ',total_comm_max_cl)

from math import *
import itertools as it


'''把可能的关联关系画出来'''
# Defining a circle that can be drawn around each community
def draw_circle_around_clique(clique,coords):
    dist=0
    temp_dist=0
    center=[0 for i in range(2)]
    color=next(colors)
    for a in clique:
        for b in clique:
            temp_dist=(coords[a][0]-coords[b][0])**2+(coords[a][1]-coords[b][1])**2
            if temp_dist>dist:
                dist=temp_dist
                for i in range(2):
                    center[i]=(coords[a][i]+coords[b][i])/2
    rad=dist**0.5/2
    cir=plt.Circle((center[0],center[1]),radius=rad*1.3,fill=False,color=color)
    plt.gca().add_patch(cir)
    plt.axis('scaled')
    return color

# Setting a cycle of colors,
global colors, hatches
colors=it.cycle('b')

# Remove "len(clique)>2" if you're interested in maxcliques with 2 edges
cliques = [clique for clique in nx.find_cliques(G) if len(clique) > 1]

# Draw the graph
nx.draw_networkx(G, pos, node_size=3, with_labels=False)
for clique in cliques:
    print("Clique to appear : ", clique)
    nx.draw_networkx_nodes(G, pos, nodelist=clique, node_color=draw_circle_around_clique(clique, pos), node_size=1000,
                           alpha=0)
plt.suptitle('Community Structure (Maximal Clique Calculation)', fontsize=22, fontname='Arial')
plt.box(on=None)
plt.axis('off')
# plt.show()

red_patch = mpatches.Patch(color='r', label='Nodes')
# blue_patch = plt.Circle(color='b', label='Overlapping Communities', fill=False)

red_patch = plt.Line2D([], [], color="red", marker='o', markerfacecolor="red", label='Nodes')
blue_patch = plt.Line2D([], [], color="blue", marker='o', markerfacecolor='white', label='Overlapping Communities')
black_patch = plt.Line2D([], [], color="black", label='Graph Edge')
plt.legend(handles=[red_patch, blue_patch, black_patch], bbox_to_anchor=(0, 1), loc='best', ncol=1)
# plt.savefig('Cliques.png', dpi=400, bbox_inches='tight')
# plt.show()

print(len(cliques))


nx.draw_networkx(G, pos,node_color='green', node_size=125, label='Participants', font_size=5)
# plt.savefig('nodes.png',dpi=400)







