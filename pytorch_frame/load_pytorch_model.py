
import pickle
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size=100

model=pickle.load(open('dnn.pkl', 'rb'))
test_dataset = torchvision.datasets.MNIST(root='data',
                                          train=False,
                                          transform=transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

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

# print('user detail images accuracy network on the test correct total cuda')
# print('scikit learn frame pytorch model neuralnet module attribute')
# print('user admin pytorch frame load pytorch model dnn pkl rb attribute version control')
# print('anaconda3 admin ')
