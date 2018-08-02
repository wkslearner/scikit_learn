import numpy as np
from desection_tree import treePlotter

class DecisionTree:
    """决策树使用方法：

        - 生成实例： clf = DecisionTrees(). 参数mode可选，ID3或C4.5，默认C4.5

        - 训练，调用fit方法： clf.fit(X,y).  X,y均为np.ndarray类型

        - 预测，调用predict方法： clf.predict(X). X为np.ndarray类型

        - 可视化决策树，调用showTree方法 

    决策过程：根据数据集计算每个变量与y值之间的信息增益，选择分割特征
             计算分割特征的每个值得信息增益，选择分割点
             按照分割点和分割特征进行分割
             然后按照这个步骤不断进行递归分割
             
    id3 和 c4.5的差异主要在于最佳分割点的参数选择，id3使用信息增益，c4.5使用信息增益比
    """

    def __init__(self, mode='C4.5'):
        self._tree = None

        if mode == 'C4.5' or mode == 'ID3':
            self._mode = mode
        else:
            raise Exception('mode should be C4.5 or ID3')


    def _calcEntropy(self, y):
        """
        函数功能：计算熵
        参数y：数据集的标签
        """
        num = y.shape[0]
        # 统计y中不同label值的个数，并用字典labelCounts存储
        labelCounts = {}
        for label in y:
            if label not in labelCounts.keys():
                labelCounts[label] = 0
            labelCounts[label] += 1
        # 计算熵
        entropy = 0.0
        for key in labelCounts:
            prob = float(labelCounts[key])/num
            entropy -= prob * np.log2(prob)    #log2(prob)是一个负值，因无人这里要使用负号

        return entropy


    def _splitDataSet(self, X, y, index, value):
        """
        函数功能：返回数据集中特征下标为index，特征值等于value的子数据集
        """
        ret = []
        featVec = X[:, index]
        X = X[:, [i for i in range(X.shape[1]) if i != index]] #求剔除index列的数据子集
        for i in range(len(featVec)):
            if featVec[i] == value:
                ret.append(i)

        return X[ret, :], y[ret]

    def _chooseBestFeatureToSplit_ID3(self, X, y):
        """ID3
        函数功能：对输入的数据集，选择最佳分割特征
        参数dataSet：数据集，最后一列为label
        主要变量说明：
                numFeatures：特征个数
                oldEntropy：原始数据集的熵
                newEntropy：按某个特征分割数据集后的熵
                infoGain：信息增益
                bestInfoGain：记录最大的信息增益
                bestFeatureIndex：信息增益最大时，所选择的分割特征的下标
        """
        numFeatures = X.shape[1]  #数据集特征数量
        oldEntropy = self._calcEntropy(y)
        bestInfoGain = 0.0
        bestFeatureIndex = -1
        # 对每个特征都计算一下infoGain，并用bestInfoGain记录最大的那个
        for i in range(numFeatures):
            featList = X[:, i]
            uniqueVals = set(featList)
            newEntropy = 0.0
            # 对第i个特征的各个value，得到各个子数据集，计算各个子数据集的熵，
            # 进一步地可以计算得到根据第i个特征分割原始数据集后的熵newEntropy
            for value in uniqueVals:
                sub_X, sub_y = self._splitDataSet(X, y, i, value)
                prob = len(sub_y) / float(len(y))
                newEntropy += prob * self._calcEntropy(sub_y)
                # 计算信息增益，根据信息增益选择最佳分割特征
            infoGain = oldEntropy - newEntropy
            #确定最好的分割点索引
            if (infoGain > bestInfoGain):
                bestInfoGain = infoGain
                bestFeatureIndex = i
        return bestFeatureIndex


    def _chooseBestFeatureToSplit_C45(self, X, y):
        """C4.5
            ID3算法计算的是信息增益，C4.5算法计算的是信息增益比，对上面ID3版本的函数稍作修改即可
        """
        numFeatures = X.shape[1]
        oldEntropy = self._calcEntropy(y)
        bestGainRatio = 0.0
        bestFeatureIndex = -1
        # 对每个特征都计算一下gainRatio=infoGain/splitInformation
        for i in range(numFeatures):
            featList = X[:, i]
            uniqueVals = set(featList)
            newEntropy = 0.0
            splitInformation = 0.0
            # 对第i个特征的各个value，得到各个子数据集，计算各个子数据集的熵，
            # 进一步地可以计算得到根据第i个特征分割原始数据集后的熵newEntropy
            for value in uniqueVals:
                sub_X, sub_y = self._splitDataSet(X, y, i, value)
                prob = len(sub_y) / float(len(y))
                newEntropy += prob * self._calcEntropy(sub_y)
                splitInformation -= prob * np.log2(prob)  #分割后的子集信息熵

            # 计算信息增益比，根据信息增益比选择最佳分割特征
            # splitInformation若为0，说明该特征的所有值都是相同的，显然不能作为分割特征

            if splitInformation == 0.0:
                pass
            else:
                infoGain = oldEntropy - newEntropy
                gainRatio = infoGain / splitInformation
                if (gainRatio > bestGainRatio):
                    bestGainRatio = gainRatio
                    bestFeatureIndex = i

        return bestFeatureIndex


    def _majorityCnt(self, labelList):
        """
        函数功能：返回labelList中出现次数最多的label
        """
        labelCount = {}
        for vote in labelList:
            if vote not in labelCount.keys():
                labelCount[vote] = 0
            labelCount[vote] += 1
        sortedClassCount = sorted(labelCount.iteritems(), key=lambda x: x[1], reverse=True)
        return sortedClassCount[0][0]


    def _createTree(self, X, y, featureIndex):
        """建立决策树
        featureIndex，类型是元组，它记录了X中的特征在原始数据中对应的下标。
        """
        labelList = list(y)
        # 所有label都相同的话，则停止分割，返回该label
        if labelList.count(labelList[0]) == len(labelList):
            return labelList[0]
        # 没有特征可分割时，停止分割，返回出现次数最多的label
        if len(featureIndex) == 0:
            return self._majorityCnt(labelList)

        # 可以继续分割的话，确定最佳分割特征
        if self._mode == 'C4.5':
            bestFeatIndex = self._chooseBestFeatureToSplit_C45(X, y)
        elif self._mode == 'ID3':
            bestFeatIndex = self._chooseBestFeatureToSplit_ID3(X, y)

        bestFeatStr = featureIndex[bestFeatIndex]  #提取最佳分割特征索引
        featureIndex = list(featureIndex)
        featureIndex.remove(bestFeatStr)
        featureIndex = tuple(featureIndex)
        # 用字典存储决策树。最佳分割特征作为key，而对应的键值仍然是一棵树（仍然用字典存储）
        myTree = {bestFeatStr: {}}
        featValues = X[:, bestFeatIndex]
        uniqueVals = set(featValues)  #分割列值去重
        for value in uniqueVals:
            #对分割列的value进行循环分割,分层不同的叶节点
            sub_X, sub_y = self._splitDataSet(X, y, bestFeatIndex, value)  #数据分割
            myTree[bestFeatStr][value] = self._createTree(sub_X, sub_y, featureIndex)
        return myTree


    def fit(self, X, y):
        #类型检查
        if isinstance(X, np.ndarray) and isinstance(y, np.ndarray):
            pass
        else:
            try:
                X = np.array(X)
                y = np.array(y)
            except:
                raise TypeError("numpy.ndarray required for X,y")

        featureIndex = tuple(['x' + str(i) for i in range(X.shape[1])])  #根据数据集列数量指定特征索引
        self._tree = self._createTree(X, y, featureIndex)
        return self  # allow chaining: clf.fit().predict()


    def predict(self, X):
        if self._tree == None:
            raise NotFittedError("Estimator not fitted, call `fit` first")

        # 检查数据类型是否为数组，否则尝试转换
        if isinstance(X, np.ndarray):
            pass
        else:
            try:
                X = np.array(X)
            except:
                raise TypeError("numpy.ndarray required for X")

        def _classify(tree, sample):
            """
            用训练好的决策树对输入数据分类 
            决策树的构建是一个递归的过程，用决策树分类也是一个递归的过程
            _classify()一次只能对一个样本（sample）分类
            To Do: 多个sample的预测怎样并行化？
            """
            featIndex = list(tree.keys())[0] #获取键值列表第一个键
            print(featIndex,'xxx')
            secondDict = tree[featIndex]
            key = sample[int(featIndex[1:])] #提取featindex值得第二位
            valueOfkey = secondDict[key]
            if isinstance(valueOfkey, dict): #判断字典实例是否可再分
                label = _classify(valueOfkey, sample)
            else:
                label = valueOfkey
            return label


        if len(X.shape) == 1:
            return _classify(self._tree, X)
        else:
            results = []
            for i in range(X.shape[0]):
                results.append(_classify(self._tree, X[i]))  #来源于fit()函数中的_tree变量

            return np.array(results)


    def predict_prob(self,X):


        pass


    def show(self):
        if self._tree == None:
            raise NotFittedError("Estimator not fitted, call `fit` first")

        # plot the tree using matplotlib

        treePlotter.createPlot(self._tree)


class NotFittedError(Exception):
    """
    Exception class to raise if estimator is used before fitting

    """
    pass