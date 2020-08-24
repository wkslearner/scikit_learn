
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import roc_curve,auc

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
input_size = 784
hidden_size = 500
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.001

print(1)
# MNIST dataset 下载
train_dataset = torchvision.datasets.MNIST(root='data',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)
print(2)
test_dataset = torchvision.datasets.MNIST(root='data',
                                          train=False,
                                          transform=transforms.ToTensor())
print(3)
# Data loader 数据载入
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
print(4)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

print(5)
# Fully connected neural network with one hidden layer
# 基于不同激活函数的DNN网络
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # 线性层
        self.relu = nn.ReLU()  # 激活函数
        self.fc2 = nn.Linear(hidden_size, num_classes)
        # hidden_size 隐藏神经元

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# 模型构建
model = NeuralNet(input_size, hidden_size, num_classes).to(device)
print(6)

# Loss and optimizer  损失函数及优化方法
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
print(7)


# Train the model  模型训练
total_step = len(train_loader)
print(total_step)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Move tensors to the configured device  把tensors加载到训练设备
        images = images.reshape(-1, 28 * 28).to(device)
        labels = labels.to(device)

        # Forward pass 正向传递
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize  反向传递及参数优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 训练过程信息展示
        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

print(8)
# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
# 不求梯度
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28 * 28).to(device)
        labels = labels.to(device)
        outputs = model(images)
        # 用softmax函数将输出结果转换成概率
        probability = torch.nn.functional.softmax(outputs, dim=1)
        # 预测结果分类
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))



print(9)
# print(model.state_dict())
# Save the model checkpoint  模型持久化
# torch.save(model.state_dict(), 'DNN-model.ckpt')

# import pickle
# pickle.dump(model, open('dnn.pkl', 'wb'))



