
'''
基于连通组件的社区发现算法
'''

from networkx import connected_components
import pandas as pd


# '''关联关系生成函数'''
# class Data(object):
#     def __init__(self, name):
#         self.__name = name
#         self.__links = set()
#
#     @property
#     def name(self):
#         return self.__name
#
#     @property
#     def links(self):
#         return set(self.__links)
#
#     #添加链接节点
#     def add_link(self, other):
#         self.__links.add(other)
#         other.__links.add(self)
#
#
# # The function to look for connected components.
# def connected_components(nodes):
#     # List of connected components found. The order is random.
#     result = []
#
#     # Make a copy of the set, so we can modify it.
#     nodes = set(nodes)
#
#     # Iterate while we still have nodes to process.
#     while nodes:
#         # Get a random node and remove it from the global set.
#         n = nodes.pop()
#         print(n)
#
#         # This set will contain the next group of nodes connected to each other.
#         group = {n}
#
#         # Build a queue with this node in it.
#         queue = [n]
#
#         # Iterate the queue.
#         # When it's empty, we finished visiting a group of connected nodes.
#         while queue:
#             # Consume the next item from the queue.
#             n = queue.pop(0)
#
#             # Fetch the neighbors.
#             neighbors = n.links  #提取邻居节点
#
#             # Remove the neighbors we already visited.
#             neighbors.difference_update(group)
#
#             # Remove the remaining nodes from the global set.
#             nodes.difference_update(neighbors)
#
#             # Add them to the group of connected nodes.
#             group.update(neighbors)
#
#             # Add them to the queue, so we visit them in the next iterations.
#             queue.extend(neighbors)
#
#         # Add the group to the list of groups.
#         result.append(group)
#
#     # Return the list of groups.
#     return result
#
#
# # The test code...
# if __name__ == "__main__":
#
#     # The first group, let's make a tree.
#     a = Data("a")
#     b = Data("b")
#     c = Data("c")
#     d = Data("d")
#     e = Data("e")
#     f = Data("f")
#     a.add_link(b)  # a
#     a.add_link(c)  # / \
#     b.add_link(d)  # b   c
#     c.add_link(e)  # /   / \
#     c.add_link(f)  # d   e   f
#
#     # The second group, let's leave a single, isolated node.
#     g = Data("g")
#
#     # The third group, let's make a cycle.
#     h = Data("h")
#     i = Data("i")
#     j = Data("j")
#     k = Data("k")
#     h.add_link(i)  # h————i
#     i.add_link(j)  # |    |
#     j.add_link(k)  # |    |
#     k.add_link(h)  # k————j
#
#     # Put all the nodes together in one big set.
#     nodes = {a, b, c, d, e, f, g, h, i, j, k}
#
#     # Find all the connected components.
#     number = 1
#     for components in connected_components(nodes):
#         names = sorted(node.name for node in components)
#         names = ", ".join(names)
#         print("Group #%i: %s" % (number, names))
#         number += 1
#
#     # You should now see the following output:
#     # Group #1: a, b, c, d, e, f
#     # Group #2: g
#     # Group #3: h, i, j, k




# Python program to print connected
# components in an undirected graph
class Graph:

    # init function to declare class variables
    def __init__(self, V):
        self.V = V
        self.adj = [[] for i in range(V)]

    #深度优先搜索算法
    def DFSUtil(self, temp, v, visited):

        # Mark the current vertex as visited
        visited[v] = True

        # Store the vertex to list
        temp.append(v)

        # Repeat for all vertices adjacent
        # to this vertex v
        for i in self.adj[v]:
            if visited[i] == False:
                # Update the list
                temp = self.DFSUtil(temp, i, visited)
        return temp
        # method to add an undirected edge

    def addEdge(self, v, w):
        self.adj[v].append(w)
        self.adj[w].append(v)

        # Method to retrieve connected components


    # in an undirected graph
    def connectedComponents(self):
        visited = []
        cc = []
        for i in range(self.V):
            visited.append(False)
        for v in range(self.V):
            if visited[v] == False:
                temp = []
                cc.append(self.DFSUtil(temp, v, visited))
        return cc

    # Driver Code


if __name__ == "__main__":
    # Create a graph given in the above diagram
    # 5 vertices numbered from 0 to 4
    g = Graph(5);
    g.addEdge(1, 0)
    g.addEdge(2, 3)
    g.addEdge(3, 4)
    cc = g.connectedComponents()
    print("Following are connected components")
    print(cc)
