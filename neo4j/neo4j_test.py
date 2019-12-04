
from py2neo import Node, Relationship,Graph

test_graph = Graph(
    "http://localhost:7474",
    username="neo4j",
    password="123456")

# 节点添加
# a = Node('Person', name='Alice')
# b = Node('Person', name='Bob')

# 节点属性添加
# a['phone']=13809889
# b['phone']=13987669

# 关系添加
# r = Relationship(a, '1', b)
# 关系属性添加
# r['num']=1

# r1 = Relationship(a, '2', b)
# r1['num']=2

# 节点关系属性联合添加
# s = a | b | r
# s1= a | b | r1

# a=Node('12',name='M2')
# a['user_id']=12
# b=Node('123',name='nomal')
# b['user_id']=123
# dist={}
# dist['12']=a
# dist['123']=b
# r1 = Relationship(dist['12'], 'phone', dist['123'])
# node1=dist['12']
# print(node1)

graph = Graph(password='123456')
# graph.create(a)
# graph.create(b)
# graph.create(r1)
graph.delete_all()

#节点创建
# a = Node("Person", name="Alice")
# b = Node("Person", name="Bob")
# ab = Relationship(a, "KNOWS", b)
# print(ab)

# 属性添加
# a['age'] = 20
# b['age'] = 21
# ab['time'] = '2017/08/31'
# print(a, b, ab)








