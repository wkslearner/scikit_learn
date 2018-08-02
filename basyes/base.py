from numpy import *

def loadDataSet():
    '''
    postingList: 进行词条切分后的文档集合
    classVec:类别标签    
    '''
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #1代表侮辱性文字，0代表正常言论

    return postingList,classVec



'''根据已有数据集构建不重复词库'''
def createVocabList(dataSet):
    vocabSet = set([])    #使用set创建不重复词表库
    for document in dataSet:
        vocabSet = vocabSet | set(document) #创建两个集合的并集
    return list(vocabSet)



'''统计输入数据集在训练数据集中的频数'''
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0]*len(vocabList)#创建一个所包含元素都为0的向量
    #遍历文档中的所有单词，如果出现了词汇表中的单词，则将输出的文档向量中的对应值设为1
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        #else: print("the word: %s is not in my Vocabulary!" % word)
    return returnVec

word_list=['dalmation', 'is','lst', 'so', 'cute', 'I', 'love', 'me','th','why','is']


'''
我们将每个词的出现与否作为一个特征，这可以被描述为词集模型(set-of-words model)。
如果一个词在文档中出现不止一次，这可能意味着包含该词是否出现在文档中所不能表达的某种信息,
这种方法被称为词袋模型(bag-of-words model)。
在词袋中，每个单词可以出现多次，而在词集中，每个词只能出现一次。
为适应词袋模型，需要对函数setOfWords2Vec稍加修改，修改后的函数称为bagOfWords2VecMN
'''

def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec



'''训练集求每个单词出现时结果为0或1的概率'''
def trainNB0(trainMatrix,trainCategory):
    '''
    朴素贝叶斯分类器训练函数(此处仅处理两类分类问题)
    trainMatrix:文档矩阵
    trainCategory:每篇文档类别标签
    '''

    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory)/float(numTrainDocs) #求概率p(y)
    #初始化所有词出现数为1，并将分母初始化为2，避免某一个概率值为0
    p0Num = ones(numWords); p1Num = ones(numWords)#
    p0Denom = 2.0; p1Denom = 2.0 #
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    #将结果取自然对数，避免下溢出，即太多很小的数相乘造成的影响
    p1Vect = log(p1Num/p1Denom)#change to log()
    p0Vect = log(p0Num/p0Denom)#change to log()
    return p0Vect,p1Vect,pAbusive



'''贝叶斯主函数'''

'''
朴素贝叶斯是使用概率论来分类的算法。其中朴素：各特征条件独立；贝叶斯：根据贝叶斯定理。

根据贝叶斯定理，对一个分类问题，给定样本特征x，样本属于类别y的概率是：

 p(y|x)=p(x|y)p(y)/p(x)-------（1）

在这里，x 是一个特征向量，设 x 维度为 M。因为朴素的假设，即特征条件独立，根据全概率公式展开，上式可以表达为：


这里，只要分别估计出，特征 Χi 在每一类的条件概率就可以了。
类别 y 的先验概率可以通过训练集算出，同样通过训练集上的统计，可以得出对应每一类上的，条件独立的特征对应的条件概率向量。 
'''
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    '''
    分类函数
    vec2Classify:要分类的向量
    p0Vec, p1Vec, pClass1:分别对应trainNB0计算得到的3个概率
    '''
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)   #目标分类概率
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)   #非目标分类概率
    if p1 > p0:
        return 1
    else:
        return 0


'''对结果进行测试'''
def testingNB():
    listOPosts,listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat=[]
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))

    #训练模型，注意此处使用array
    p0V,p1V,pAb = trainNB0(array(trainMat),array(listClasses))

    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb))
    testEntry = ['stupid', 'garbage','yes','sb','my','is']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb))


testingNB()