
'''
图片识别
https://www.pluralsight.com/guides/image-classification-with-pytorch
'''

import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
# %matplotlib inline  # 在jupyter 中 画图时 需要添加的点
import os

os.getcwd()
# place the files in your IDE working dicrectory .
labels = pd.read_csv(r'/Users/admin/Downloads/aerial-cactus-identification/train.csv')
submission = pd.read_csv(r'/Users/admin/Downloads/aerial-cactus-identification/sample_submission.csv')

train_path = r'/Users/admin/Downloads/aerial-cactus-identification/train'
test_path = r'/Users/admin/Downloads/aerial-cactus-identification/test'

labels['has_cactus'].value_counts()
label = 'Has Cactus', 'Hasn\'t Cactus'
# plt.figure(figsize = (8,8))
# plt.pie(labels.groupby('has_cactus').size(), labels = label, autopct='%1.1f%%', shadow=True, startangle=90)
# plt.show()

# 生成图像组合页面
import matplotlib.image as img
fig,ax = plt.subplots(1,5,figsize = (15,3))
for i,idx in enumerate(labels[labels['has_cactus'] == 1]['id'][-5:]):
    path = os.path.join(train_path,idx)
    ax[i].imshow(img.imread(path))

fig,ax = plt.subplots(1,5,figsize = (15,3))
for i,idx in enumerate(labels[labels['has_cactus'] == 0]['id'][:5]):
    path = os.path.join(train_path,idx)
    ax[i].imshow(img.imread(path))

# plt.show()

import numpy as np
import matplotlib.pyplot as plt
def imshow(image, ax=None, title=None, normalize=True):
    if ax is None:
        fig, ax = plt.subplots()
    image = image.numpy().transpose((1, 2, 0))

    if normalize:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)

    ax.imshow(image)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', length=0)
    ax.set_xticklabels('')
    ax.set_yticklabels('')

    return ax

# 图片信息获取（存储的数据结构）
class CactiDataset(Dataset):
    def __init__(self, data, path, transform=None):
        super().__init__()
        self.data = data.values
        self.path = path
        self.transform = transform

    # 数据长度
    def __len__(self):
        return len(self.data)

    # 标签属性
    def __getitem__(self, index):
        img_name, label = self.data[index]
        img_path = os.path.join(self.path, img_name)
        image = img.imread(img_path)
        if self.transform is not None:
            image = self.transform(image)
        return image, label

# 数据转换函数
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
train_transform = transforms.Compose([transforms.ToPILImage(),
                                      transforms.ToTensor(),
                                      normalize])

test_transform = transforms.Compose([transforms.ToPILImage(),
                                     transforms.ToTensor(),
                                     normalize])

valid_transform = transforms.Compose([transforms.ToPILImage(),
                                     transforms.ToTensor(),
                                     normalize])

train, valid_data = train_test_split(labels, stratify=labels.has_cactus, test_size=0.2)
# print(train)

# 生成指定结构数据
train_data = CactiDataset(train, train_path, train_transform )
valid_data = CactiDataset(valid_data, train_path, valid_transform)
test_data = CactiDataset(submission, test_path, test_transform )
# print(train_data)
print(train_data)


# Hyper parameters
num_epochs = 35
num_classes = 2
batch_size = 25
learning_rate = 0.001

# CPU or GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# 生成数据迭代器（方便训练）
# batch_size 每个批次数据长度 ， shuffle 是否打乱数据
train_loader = DataLoader(dataset = train_data, batch_size = batch_size, shuffle=True, num_workers=0)
valid_loader = DataLoader(dataset = valid_data, batch_size = batch_size, shuffle=False, num_workers=0)
test_loader = DataLoader(dataset = test_data, batch_size = batch_size, shuffle=False, num_workers=0)

import numpy as np
import matplotlib.pyplot as plt

def imshow(image, ax=None, title=None, normalize=True):
    if ax is None:
        fig, ax = plt.subplots()
    image = image.numpy().transpose((1, 2, 0))

    if normalize:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)

    ax.imshow(image)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', length=0)
    ax.set_xticklabels('')
    ax.set_yticklabels('')

    return ax

trainimages, trainlabels = next(iter(train_loader))
fig, axes = plt.subplots(figsize=(12, 12), ncols=5)
print('training images')
for i in range(5):
    axe1 = axes[i]
    imshow(trainimages[i], ax=axe1, normalize=False)

print(trainimages[0].size())

epochs = 35
batch_size = 25
learning_rate = 0.001
import torch
import torch.nn as nn
import torch.nn.functional as F

# 卷积神经网络主函数
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=3)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=3)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(720, 1024)
        self.fc2 = nn.Linear(1024, 2)

    # 前馈网络
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))  # 池化层
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(x.shape[0],-1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x

model = CNN()
print(model)

# 建模函数
model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr = learning_rate)

# % % time
# keeping-track-of-losses
train_losses = []
valid_losses = []
"""
# 模型训练
for epoch in range(1, num_epochs + 1):
    # keep-track-of-training-and-validation-loss
    train_loss = 0.0
    valid_loss = 0.0

    # training-the-model
    model.train()
    for data, target in train_loader:
        # move-tensors-to-GPU
        data = data.to(device)
        target = target.to(device)

        # clear-the-gradients-of-all-optimized-variables
        optimizer.zero_grad()
        # forward-pass: compute-predicted-outputs-by-passing-inputs-to-the-model
        output = model(data)
        # calculate-the-batch-loss
        loss = criterion(output, target)
        # backward-pass: compute-gradient-of-the-loss-wrt-model-parameters
        loss.backward()
        # perform-a-ingle-optimization-step (parameter-update)
        optimizer.step()
        # update-training-loss
        train_loss += loss.item() * data.size(0)

    # validate-the-model
    model.eval()
    for data, target in valid_loader:
        data = data.to(device)
        target = target.to(device)
        output = model(data)
        loss = criterion(output, target)

        # update-average-validation-loss
        valid_loss += loss.item() * data.size(0)

    # calculate-average-losses
    train_loss = train_loss / len(train_loader.sampler)
    valid_loss = valid_loss / len(valid_loader.sampler)
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)

    # print-training/validation-statistics
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        epoch, train_loss, valid_loss))

"""
# 基于以保存模型参数文件读取
model.load_state_dict(torch.load('model.ckpt'))

# 模型测试
# test-the-model
model.eval()  # it-disables-dropout
pred_result=[]
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in valid_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        # print(outputs.data)
        _, predicted = torch.max(outputs.data,1)
        print('pred',predicted)
        total += labels.size(0)
        print('label',labels)
        correct += (predicted == labels).sum().item()
        # pred_result.append(predicted,labels)

    print('Test Accuracy of the model: {} %'.format(100 * correct / total))


# Save
# torch.save(model.state_dict(), 'model.ckpt')

# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'

# plt.plot(train_losses, label='Training loss')
# plt.plot(valid_losses, label='Validation loss')
# plt.xlabel("Epochs")
# plt.ylabel("Loss")
# plt.legend(frameon=False)
# plt.show()










