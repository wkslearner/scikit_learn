''''''
import numpy as np
import pandas as pd
import networkx as nx

def _plain_bfs(G, source):
    """A fast BFS node generator"""
    G_adj = G.adj
    print(G_adj)
    seen = set()
    nextlevel = {source}
    while nextlevel:
        thislevel = nextlevel
        nextlevel = set()
        for v in thislevel:
            if v not in seen:
                yield v
                seen.add(v)
                nextlevel.update(G_adj[v])

def connected_components(G):
    seen = set()
    for v in G:
        if v not in seen:
            c = set(_plain_bfs(G, v))
            # print(c)
            yield c
            seen.update(c)

dfs=pd.DataFrame({'a':[1,2,3,4,5,6],'b':[3,3,4,9,6,8]})
# print(dfs)
G=nx.from_pandas_edgelist(dfs,'a','b')
# com=nx.algorithms.connected_components(G)
# print(com)

com=connected_components(G)
for co in com:
    print(co)
# print(com)





