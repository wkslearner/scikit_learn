
'''
基于自编码神经网络的欺诈检测

autoencoder 原理介绍
https://www.cnblogs.com/bonelee/p/9957276.html
'''

from keras.layers import Input, Dense
from keras.models import Model, Sequential
from keras import regularizers
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.manifold import TSNE
from sklearn import preprocessing
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import os

os.environ['KMP_DUPLICATE_LIB_OK'] ='True'
sns.set(style="whitegrid")
np.random.seed(203)

data = pd.read_csv("creditcard.csv")

data["Time"] = data["Time"].apply(lambda x : x / 3600 % 24)
print(data.shape)

#对数据进行聚合统计
vc = data['Class'].value_counts().to_frame().reset_index()
vc['percent'] = vc["Class"].apply(lambda x : round(100*float(x) / len(data), 2))
vc = vc.rename(columns = {"index" : "Target", "Class" : "Count"})

#对数据集进行随机抽取
non_fraud = data[data['Class'] == 0].sample(1000)
fraud = data[data['Class'] == 1]

#生成训练数据集
df = non_fraud.append(fraud).sample(frac=1).reset_index(drop=True)
X = df.drop(['Class'], axis = 1).values
Y = df["Class"].values

'''展示结果'''
def tsne_plot(x1, y1, name="graph.png"):
    tsne = TSNE(n_components=2, random_state=0)
    X_t = tsne.fit_transform(x1)

    plt.figure(figsize=(12, 8))
    plt.scatter(X_t[np.where(y1 == 0), 0], X_t[np.where(y1 == 0), 1], marker='o', color='g', linewidth='1', alpha=0.8,
                label='Non Fraud')
    plt.scatter(X_t[np.where(y1 == 1), 0], X_t[np.where(y1 == 1), 1], marker='o', color='r', linewidth='1', alpha=0.8,
                label='Fraud')

    plt.legend(loc='best');
    plt.savefig(name);
    plt.show();

# tsne_plot(X, Y, "original.png")

## input layer 输入层
input_layer = Input(shape=(X.shape[1],))
print(input_layer)

## encoding part  自编码
encoded = Dense(100, activation='tanh', activity_regularizer=regularizers.l1(10e-5))(input_layer)
encoded = Dense(50, activation='relu')(encoded)

## decoding part 解码
decoded = Dense(50, activation='tanh')(encoded)
decoded = Dense(100, activation='tanh')(decoded)

## output layer 输出层
output_layer = Dense(X.shape[1], activation='relu')(decoded)

# 模型构建
autoencoder = Model(input_layer, output_layer)

# 模型编译
autoencoder.compile(optimizer="adadelta", loss="mse")

# 用整体数据作为验证集
x = data.drop(["Class"], axis=1)
y = data["Class"].values

# 数据标准化
x_scale = preprocessing.MinMaxScaler().fit_transform(x.values)
# 数据类别
x_norm, x_fraud = x_scale[y == 0], x_scale[y == 1]

# 模型训练
autoencoder.fit(x_norm[0:2000], x_norm[0:2000],
                batch_size = 256, epochs = 10,
                shuffle = True, validation_split = 0.20);

# sequential模型构造器
hidden_representation = Sequential()
hidden_representation.add(autoencoder.layers[0])
hidden_representation.add(autoencoder.layers[1])
hidden_representation.add(autoencoder.layers[2])

# 使用autoencoder 进行数据降维
norm_hid_rep = hidden_representation.predict(x_norm[:3000])  # 正常的前3000条记录预测
fraud_hid_rep = hidden_representation.predict(x_fraud)  # 欺诈部分数据预测

rep_x = np.append(norm_hid_rep, fraud_hid_rep, axis = 0)
y_n = np.zeros(norm_hid_rep.shape[0])  #
y_f = np.ones(fraud_hid_rep.shape[0])
rep_y = np.append(y_n, y_f)

# tsne_plot(rep_x, rep_y, "latent_representation.png")
# norm_pred=autoencoder.predict(x_norm[:3000])
# fraud_pred=autoencoder.predict(x_fraud)
# # print(norm_pred)
# # print(fraud_pred)
# pred_y=np.append(norm_pred,fraud_pred)
# print(pred_y.shape)
# print(rep_y.shape)
# # print(fraud_pred)
# # print(accuracy_score(pred_y,rep_y))
# print(accuracy_score(pred_y,rep_y))

# 特征映射后使用逻辑回归建模
train_x, val_x, train_y, val_y = train_test_split(rep_x, rep_y, test_size=0.25)
clf = LogisticRegression(solver="lbfgs").fit(train_x, train_y)
pred_y = clf.predict(val_x)
print ("")
print ("Classification Report: ")
print (classification_report(val_y, pred_y))
print ("")
print ("Accuracy Score: ", accuracy_score(val_y, pred_y))



