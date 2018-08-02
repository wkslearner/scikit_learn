import numpy as np
from sklearn import datasets

'''单隐藏层神经网络'''
# sigmoid function
# deriv=ture 是求的是导数
'''
def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

# input dataset
X = np.array([  [0,0,1],
                [1,1,1],
                [1,0,1],
                [0,1,1] ])

# output dataset
y = np.array([[0,1,1,0]]).T

# seed random numbers to make calculation
np.random.seed(1)

# initialize weights randomly with mean 0
syn0 = 2*np.random.random((3,1)) - 1
print(syn0)

# 迭代次数
for iter in range(10000):

    # forward propagation 前向传播
    # l0也就是输入层
    l0 = X
    l1 = nonlin(np.dot(l0,syn0))

    # how much did we miss? 实际结果和l1层误差
    l1_error = y - l1

    # multiply how much we missed by the
    # slope of the sigmoid at the values in l1
    l1_delta = l1_error * nonlin(l1,True)

    # update weights
    syn0 += np.dot(l0.T,l1_delta)

print ("Output After Training:")
print (l1)


def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

X = np.array([[0,0,1],
            [0,1,1],
            [1,0,1],
            [1,1,1]])

y = np.array([[0],[1],[1],[0]])

np.random.seed(1)

# randomly initialize our weights with mean 0
syn0 = 2*np.random.random((3,5)) - 1
syn1 = 2*np.random.random((5,1)) - 1

for j in range(60000):

    # Feed forward through layers 0, 1, and 2
    l0 = X
    l1 = nonlin(np.dot(l0,syn0))
    l2 = nonlin(np.dot(l1,syn1))

    # how much did we miss the target value? 输出层和实际y的差异
    l2_error = y - l2

    if (j% 10000) == 0:
        print ("Error:" + str(np.mean(np.abs(l2_error))))

    # in what direction is the target value?
    # were we really sure? if so, don't change too much.
    l2_delta = l2_error*nonlin(l2,deriv=True)

    # how much did each l1 value contribute to the l2 error (according to the weights)?
    l1_error = l2_delta.dot(syn1.T)

    # in what direction is the target l1?
    # were we really sure? if so, don't change too much.
    l1_delta = l1_error * nonlin(l1,deriv=True)

    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)
    
print (l2)
'''

iris=datasets.load_iris()
iris_data=iris.data
iris_y=iris.target
iris_col=iris.feature_names


#激活函数
def tanh(x):
    return np.tanh(x)

def tanh_deriv(x):
    return 1.0 - np.tanh(x)*np.tanh(x)

def logistic(x):
    return 1/(1 + np.exp(-x))

def logistic_derivative(x):
    return logistic(x)*(1-logistic(x))


class NeuralNetwork:
    def __init__(self, layers, activation='tanh'):
        """
        :param layers: A list containing the number of units in each layer.
        Should be at least two values
        :param activation: The activation function to be used. Can be
        "logistic" or "tanh"
        """

        #设定激活函数
        if activation == 'logistic':
            self.activation = logistic
            self.activation_deriv = logistic_derivative
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_deriv = tanh_deriv

        #初始化随机权重
        self.weights = []
        for i in range(1, len(layers) - 1):
            self.weights.append((2*np.random.random((layers[i - 1] + 1, layers[i] + 1))-1)*0.25)
            self.weights.append((2*np.random.random((layers[i] + 1, layers[i + 1]))-1)*0.25)


    def fit(self, X, y, learning_rate=0.2, epochs=10000):
        #数组转换成至少2维
        X = np.atleast_2d(X)

        #生成临时数组
        temp = np.ones([X.shape[0], X.shape[1]+1])
        temp[:, 0:-1] = X
        X = temp
        y = np.array(y)

        for k in range(epochs):
            #随机选取从训练集中选取一行
            i = np.random.randint(X.shape[0])
            a = [X[i]]

            for l in range(len(self.weights)):
                a.append(self.activation(np.dot(a[l], self.weights[l])))

            #计算训练误差
            error = y[i] - a[-1]
            #计算参数deltas
            deltas = [error * self.activation_deriv(a[-1])]

            #权重转置和函数导数的积
            for l in range(len(a) - 2, 0, -1):
                deltas.append(deltas[-1].dot(self.weights[l].T)*self.activation_deriv(a[l]))
            deltas.reverse()

            #权重更新
            for i in range(len(self.weights)):
                layer = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i])
                self.weights[i] += learning_rate * layer.T.dot(delta)


    def predict(self, x):
        x = np.array(x)
        temp = np.ones(x.shape[0]+1)
        temp[0:-1] = x
        a = temp
        for l in range(0, len(self.weights)):
            print(self.weights[l])
            a = self.activation(np.dot(a, self.weights[l]))
        return a


nn = NeuralNetwork([4,2,1], 'logistic')
x = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([1,0,0,1])


#训练模型
nn.fit(iris_data[:100],iris_y[:100],0.1,10000)

for i in iris_data[100:]:
    print(i, nn.predict(i))




