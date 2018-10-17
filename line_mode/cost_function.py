
import numpy as np


def sigmoid(z):
    return 1/(1+np.exp(-z))


'''逻辑回归损失函数'''
def logit_costfunc(theta,x,y,leraningrate):
    '''
    :param theta: 
    :param x: 特征矩阵
    :param y: 目标变量
    :return: 
    '''

    #logit回归损失函数公式 𝐶𝑜𝑠𝑡(ℎ𝜃(𝑥), 𝑦) =−𝑦×𝑙𝑜𝑔(ℎ𝜃(𝑥))−(1−𝑦)×𝑙𝑜𝑔(1−ℎ𝜃(𝑥))

    theta=np.matrix(theta)

    x=np.matrix(x)
    y=np.matrix(y)

    first=np.multiply(-y,np.log(sigmoid(x*theta.T)))
    second=np.multiply((1-y),np.log(1-sigmoid(x*theta.T)))
    #损失函数正则化部分
    reg=(leraningrate/(2*len(x))*np.sum(np.power(theta[:,1:theta.shape[1]],2)))

    return np.sum(first-second)/(len(x))+reg


