import random

from sklearn import model_selection


def UpdateWeights(real,pred, w0, alpha):
    '''
    :param real: 真实值
    :param pred: 预测值
    :param w0: 上一步的样本权重
    :param alpha: 这次的alpha值
    :return: 更新后的样本权重
    '''
    if abs(real-pred)<0.0001:
        w1 = w0*np.exp(-alpha)
    else:
        w1 = w0*np.exp(alpha)
    return w1


def  TrainAda(df, feature_list, iteration, y,featureSampling=False, treeDepth=3):
    '''
    :param df:训练集，包含特征和标签
    :param feature_list:特征集
    :param iteration: 迭代次数
    :param featureSampling: 从特征集中的抽样率。默认不进行特征抽样
    :param treeDepth每棵树的深度
    :return:每次迭代的权重和CART模型
    '''
    N = df.shape[0]
    w0 = np.array([1.0]*N)/N
    alpha_list = []
    cart_list = []
    df2 = df.copy()
    df2['w0'] = w0
    for k in range(iteration):
        if not featureSampling:
            subFeatures = feature_list
        else:
            samplingSize = int(len(feature_list)*featureSampling)
            subFeatures = random.sample(feature_list, samplingSize)
        cart = TrainCART(df2,subFeatures,y,'w0',treeDepth)
        df2['pred'] = df2[subFeatures].apply(lambda x: predCART(x, cart), axis=1)
        err_pred = df2.apply(lambda x: abs(x[y] - x.pred), axis=1)
        err = sum(err_pred)*1.0/N
        if err >= 0.5:   #预测错误率超过0.5的分类器进行舍弃
            continue
        alpha = 0.5*np.log((1-err)/err)
        cart_list.append(cart)
        alpha_list.append(alpha)
        w2 = df2.apply(lambda x: UpdateWeights(x[y], x.pred, x.w0, alpha),axis = 1)
        w0 = [i/sum(w2) for i in w2]
        w_update = max(np.abs(w2 - df2['w0']))
        df2['w0'] = w0
        if err<=0.00001:
            break

        #print '{} step:'.format(k), 'Error rate is {}, '.format(err), 'alpha is {}, '.format(alpha), 'w update is {}'.format(w_update)
        print ('{0} step: Error rate is {1}, alpha is {2}, w update is {3}'.format(k,err,alpha, w_update))
    return {'alpha':alpha_list, 'cart model':cart_list}


'''导入数据集并处理'''
AllData = pd.read_csv('bank.csv', header=0)
Attributes = list(AllData.columns)
Attributes.remove('y')
AllData['y2'] = AllData['y'].apply(lambda x: int(x == 'yes'))
smallData = AllData[Attributes+['y2']]
#fewerAttributes = [u'age', u'job', u'marital', u'education', u'default', u'cash_loan', u'contact_number_type', u'maturity']

df1 = smallData.loc[smallData['y2'] == 1]
df0 = smallData.loc[smallData['y2'] == 0]
df0_2 = df0.iloc[0:1000]
myData= pd.concat([df0_2,df1])


X_train,X_test = model_selection.train_test_split(myData,test_size=0.4, random_state=0)
x1 = X_train.copy()
x2 = X_test.copy()


'''进行cart模型训练，深度为3-9层'''
cart_list = []
for depth in range(3,10):

    x1 = X_train.copy()
    x2 = X_test.copy()
    x1['w'] = [1.0/x1.shape[0]]*x1.shape[0]  #构造训练集权重列
    cart = TrainCART(x1,Attributes,'y2','w',depth)  #训练模型
    cart_list.append(cart)
    x2['y3'] = x2[Attributes].apply(lambda x: predCART(x, cart), axis=1)  #跑测试集数据
    err_pred = x2.apply(lambda x: abs(x.y2 - x.y3), axis=1)
    err = sum(err_pred)*1.0/len(err_pred)
    print (depth, err)




"""
3 0.387520525452
4 0.410509031199
5 0.343185550082
6 0.454844006568
7 0.487684729064
8 0.422003284072
9 0.353037766831
"""


'''
T = 5
AdaModel = TrainAda(X_train, Attributes, T, y,treeDepth=3)

alpha = AdaModel['alpha']
cart_list = AdaModel['cart model']
pred = X_test[Attributes].apply(lambda x: 2*(predCART(x, cart_list[0])-1)*alpha[0], axis=1)
for k in range(1,len(cart_list)):
    # 需要把｛0，1｝变成｛－1，1｝，再做预测。因为最后的结果，是加权后的sign值
    pred2 = X_test[Attributes].apply(lambda x: 2*(predCART(x, cart_list[k])-1)*alpha[k], axis=1)
    pred = pd.concat([pred,pred2],axis=1)
X_test['ada'] = pred.apply(sum,axis=1)
X_test['ada'] = X_test['ada'].map(lambda x: int(x>0))  #再把｛－1，1｝映射成｛0，1｝
err_pred = X_test.apply(lambda x: abs(x.y2 - x.ada), axis=1)
err = sum(err_pred)*1.0/len(err_pred)
print (err) #  0.35139573070607555

'''