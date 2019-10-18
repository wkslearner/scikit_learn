
import numpy as np

class DecisionTree:

    def __init__(self,mode='id3'):
        self._tree=None

        if mode=='id3' or mode=='c4.5':
            self._mode=mode
        else:
            raise Exception('mode should be id3 or c4.5')


    def _calentropy(self,y):
        num = y.shape[0]
        # 统计y中不同label值的个数，并用字典labelCounts存储
        labelCounts = {}
        for label in y:
            if label not in labelCounts.keys(): labelCounts[label] = 0
            labelCounts[label] += 1
        # 计算熵
        entropy = 0.0
        for key in labelCounts:
            prob = float(labelCounts[key]) / num
            entropy -= prob * np.log2(prob)
        return entropy



    def _splitDataSet(self, X, y, index, value):
        """
        函数功能：返回数据集中特征下标为index，特征值等于value的子数据集
        """
        ret = []
        featVec = X[:, index]
        X = X[:, [i for i in range(X.shape[1]) if i != index]]
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

        numFeatures = X.shape[1]
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
            if (infoGain > bestInfoGain):
                bestInfoGain = infoGain
                bestFeatureIndex = i

        return bestFeatureIndex





x = [[1, 2, 0, 1, 0],
     [0, 1, 1, 0, 1],
     [1, 0, 0, 0, 1],
     [2, 1, 1, 0, 1],
     [1, 1, 0, 1, 1]]

y = ['yes', 'yes', 'no', 'no', 'no']

TREE=DecisionTree(mode='id3')
TREE.fit(x,y)

