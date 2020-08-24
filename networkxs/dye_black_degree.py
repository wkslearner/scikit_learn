'''
染黑度算法
'''

class DyeBlackDegree():

    def __init__(self,Graph,weight,node_status):
        self.Graph=Graph
        self.weight=weight
        self.node_status=node_status  #字典 逾期值为1 未逾期为0

    '''图中所有点的染黑度'''
    def dye_black_all(self):
        dye_black_dict={}
        for node in self.Graph.nodes:
            dye_black_degree=self.dye_black(node)
            dye_black_dict[node]=dye_black_degree

        return dye_black_dict


    '''单点染黑度'''
    def dye_black(self,node):
        view_node = []
        degree_result_list = []
        degree_list = list(self.Graph[node])
        view_node.append(node)

        # 每一层包含的点
        while len(degree_list) > 0:
            degree_result_list.append(list(degree_list))
            degree_list, view_node = self.degree_func(self.Graph, degree_list, view_node)
            view_node = view_node

        # 路径长度
        path_len = len(degree_list)
        weight_list = self.weight_distribute(path_len)
        overrate_list = self.overrate_caculate(degree_result_list)

        dye_black_degree = 0
        for weight, overrate in zip(weight_list, overrate_list):
            dye_black_degree += weight * overrate

        return dye_black_degree

    '''关系层逾期率计算'''
    def overrate_caculate(self,degree_list):
        overrate_list=[]
        for node_list in degree_list:
            list_len = len(node_list)
            sum_over = 0
            for node in node_list:
                node_sta=self.node_status[node]
                sum_over+=node_sta

            overrate=sum_over/list_len
            overrate_list.append(overrate)

        return overrate_list


    '''权重分配'''
    def weight_distribute(self,path_len):
        if path_len==1:
            return [1]
        else:
            weight_list=[]
            weight_now = self.weight
            weight_user= self.weight
            weight_remain = 1 - weight_user
            weight_list.append(weight_now)
            for i in range(path_len-2):
                weight_now=weight_remain*self.weight
                weight_user = weight_user + weight_now
                weight_remain = 1 - weight_user
                # print(weight_now, weight_user, weight_remain)
                weight_list.append(weight_now)

            weight_list.append(weight_remain)

            return weight_list


    '''计算节点外延每层数据点'''
    def degree_func(self,Graph, node_list, view_node=[]):
        result_list = []
        for node in node_list:
            node_degree = Graph[node]
            view_node.append(node)
            for nodes in list(node_degree):
                # print(Graph[nodes])
                result_list += list(Graph[nodes])

        result_list = set(result_list)
        result_list = result_list - set(view_node)

        return result_list, view_node


import networkx as nx
G = nx.barabasi_albert_graph(10,3)  #生成一个n=1000，m=3的BA无标度网络
dye=DyeBlackDegree(G,0.7)
dye.weight_distribute(5)







