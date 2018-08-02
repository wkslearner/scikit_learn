import numpy as np
import pandas as pd
from numpy import dot
from numpy.linalg import inv
from sklearn.datasets import load_iris
import pylab as pl
import statsmodels.api as sm
import sklearn
import copy


#http://cdn.powerxing.com/files/lr-binary.csv
#http://www.ats.ucla.edu/stat/data/binary.csv
df = pd.read_csv("http://cdn.powerxing.com/files/lr-binary.csv")
df.columns=['admit','gre','gpa','grp']
dummy = pd.get_dummies(df['grp'],prefix='grp')

df_1=df[['admit','gre','gpa']]
df_2=dummy.ix[:,'grp_2':]
new_df=pd.concat([df_1,df_2],axis=1)
new_df['intercept']=1.0
train_data=new_df.columns[1:]


logit=sm.Logit(new_df['admit'],new_df[train_data])
result=logit.fit()


predict_data=copy.deepcopy(new_df)
predict_col=predict_data.columns[1:]
predict_data['predict']=result.predict(predict_data[predict_col])
print(predict_data)

total = 0
hit = 0
for value in predict_data.values:
    # 预测分数 predict, 是数据中的最后一列
    predict = value[-1]
    # 实际录取结果
    admit = int(value[0])


    # 假定预测概率大于0.5则表示预测被录取
    if predict > 0.3:
        total += 1
        # 表示预测命中
        if admit == 1:
            hit += 1



# 输出结果
print ('Total: %d, Hit: %d, Precision: %.2f' % (total, hit, 100.0 * hit / total) )
# Total: 49, Hit: 30, Precision: 61.22



#查看数据基本信息
#print(iris.describe())   #数据统计描述
#print (iris.std())

#绘图观测
#iris.hist()
#pl.show()


#dummy = pd.get_dummies(iris['Species']) # 对Species生成哑变量
#print(dummy)
#iris = pd.concat([iris, dummy], axis =1 )
#train_data=iris.columns[1:5]
#train_data=train_data.drop('Species')
#print(iris[train_data])
#logit=sm.Logit(iris['Species'],iris[train_data])

#print(logit.fit())


'''
iris = iris.iloc[0:100, :] # 截取前一百行样本


# 构建Logistic Regression , 对Species是否为setosa进行分类 setosa ~ Sepal.Length
# Y = g(BX) = 1/(1+exp(-BX))
def logit(x):
    return np.longfloat(1.0/(1+np.exp(-x)))

temp = pd.DataFrame(iris.iloc[:, 0])
temp['x0'] = 1.0
X = temp.iloc[:,[1,0]]
Y = np.reshape(iris['setosa'],(len(iris),1)) #整理出X矩阵 和 Y矩阵


# 批量梯度下降法
m,n = X.shape #矩阵大小
alpha = 0.0065 #设定学习速率
theta_g = np.zeros((n,1)) #初始化参数
maxCycles = 3000 #迭代次数
J = pd.Series(np.arange(maxCycles, dtype = float)) #损失函数


for i in range(maxCycles):
    h = logit(dot(X, theta_g)) #估计值
    J[i] = -(1/100.)*np.sum(Y*np.log(h)+(1-Y)*np.log(1-h)) #计算损失函数值
    error = h - Y #误差
    grad = dot(X.T, error) #梯度
    theta_g -= alpha * grad


print (theta_g)
print (J.plot())


# 牛顿方法
theta_n = np.zeros((n,1)) #初始化参数
maxCycles = 10 #迭代次数
C = pd.Series(np.arange(maxCycles, dtype = float)) #损失函数
for i in range(maxCycles):
    h = logit(dot(X, theta_n)) #估计值
    C[i] = -(1/100.)*np.sum(Y*np.log(h)+(1-Y)*np.log(1-h)) #计算损失函数值
    error = h - Y #误差
    grad = dot(X.T, error) #梯度
    A =  h*(1-h)* np.eye(len(X))
    H = np.mat(X.T)* A * np.mat(X) #海瑟矩阵, H = X`AX
    H=H.astype(float)
    theta_n -= inv(H)*grad

print (theta_n)
print (C.plot())

'''