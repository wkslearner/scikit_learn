
'''
基于pytorch 的逻辑回归模型
'''

import torch
from torch.autograd import Variable
from torch.nn import functional as F
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


iris=load_iris()
# 转化为tensor
# origin_data=torch.Tensor(iris.data)
# target_data=torch.Tensor(iris.target)

# 数据转换成tensor
def turn_tensor(data_list):
    result_list=[]
    for data in data_list:
        tensor_data=Variable(torch.Tensor(data))
        result_list.append(tensor_data)

    return result_list

x_train,x_test,y_train,y_test=turn_tensor(train_test_split(iris.data,iris.target,test_size=0.2,random_state=0))
print(x_train.__class__)

# x_data = Variable(torch.Tensor([[10.0], [9.0], [3.0], [2.0]]))
# y_data = Variable(torch.Tensor([[90.0], [80.0], [50.0], [30.0]]))


# 二分类逻辑回归
class LinearRegression(torch.nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = torch.nn.Linear(4, 1)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

# model = LinearRegression()
# criterion = torch.nn.MSELoss(size_average=False)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
#
# for epoch in range(20):
#     model.train()
#     optimizer.zero_grad()
#     # Forward pass
#     y_pred = model(x_train)
#     # Compute Loss
#     loss = criterion(y_pred, y_train)
#     # Backward pass
#     loss.backward()
#     optimizer.step()
#
# new_x = Variable(torch.Tensor([[4.0]]))
# y_pred = model(x_test)
# print("predicted Y value: ", y_pred.data[0])

# 多分类逻辑回归
class LogisticRegression(torch.nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(4, 1)

    def forward(self, x):
        y_pred = F.sigmoid(self.linear(x))
        return y_pred

model = LogisticRegression()
criterion = torch.nn.MSELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(20):
    model.train()
    optimizer.zero_grad()
    # Forward pass
    y_pred = model(x_train)
    # Compute Loss
    loss = criterion(y_pred, y_train)
    # Backward pass
    loss.backward()
    optimizer.step()

y_pred = model(x_test)
print("predicted Y value: ", y_pred.data)


'''

'''

