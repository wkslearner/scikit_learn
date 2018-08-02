# -*- coding: utf-8 -*-
import numpy as np
from sklearn import neighbors
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import pandas as pd

iris = load_iris()
data = iris.data
target = iris.target

end_list = []
for i in range(len(data)):
    row = list(data[i])
    row.append(target[i])
    end_list.append(row)

df = pd.DataFrame(end_list, columns=['var1', 'var2', 'var3', 'var4', 'target'])
df = df[df['target'] < 2]
x = df[['var1', 'var2', 'var3', 'var4']]
y = df['target']

''''' 拆分训练数据与测试数据 '''
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

''' 创建网格以方便绘制 '''
'''
h = .01
x_min, x_max = x[:, 0].min() - 0.1, x[:, 0].max() + 0.1
y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

'''

''''' 训练KNN分类器 '''
clf = neighbors.KNeighborsClassifier(algorithm='kd_tree')
clf.fit(x_train, y_train)

'''''测试结果的打印'''
answer = clf.predict(x)
# print(x)
print(answer)
# print(y)
print(np.mean(answer == y))


'''''准确率与召回率'''
precision, recall, thresholds = precision_recall_curve(y_train, clf.predict(x_train))
answer = clf.predict_proba(x)[:, 1]
print(classification_report(y, answer, target_names=['thin', 'fat']))

''''' 将整个测试空间的分类结果用不同颜色区分开'''

'''
answer = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
z = answer.reshape(xx.shape)
plt.contourf(xx, yy, z, cmap=plt.cm.Paired, alpha=0.8)
'''

''''' 绘制训练样本 '''

# plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap=plt.cm.Paired)
# plt.xlabel(u'身高')
# plt.ylabel(u'体重')
# plt.show()
