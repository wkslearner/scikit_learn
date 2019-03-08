
'''
用户收货地址相似度分析
5.优化方向
1.是否需要加停用词过滤？
2.是否需要提取地址关键词？
'''

import jieba
from jieba import posseg
from gensim import corpora,models,similarities
import pandas as pd
import numpy as np


#dataset=pd.read_excel('/Users/admin/Documents/data_analysis/fraud_model/同一地址多订单名单.xlsx')
dataset=pd.read_excel('/Users/admin/Documents/data_analysis/fraud_model/data_source/fraud_data.xlsx')
dataset['address']=dataset['address'].fillna('null')
all_doc=dataset['address']


# 1 分词
# 1.1 历史比较文档的分词
all_doc_list = []
for doc in all_doc:
    doc=doc.replace(' ', '')
    doc_list = [word for word in jieba.cut_for_search(doc)]
    # doc_list = [word for word in jieba.cut(doc)]
    all_doc_list.append(doc_list)


# 1.2 测试文档的分词
doc_test="湖北省武汉  洪山区书城路名士一号3栋1509号3 栋1509号 "
doc_test=doc_test.replace(' ', '')
print(doc_test)
doc_test_list = [word for word in jieba.cut_for_search(doc_test)]


#获取分词词性
print([(x.word,x.flag) for x in posseg.cut(doc_test)])


# 2 制作语料库
# 2.1 获取词袋
dictionary = corpora.Dictionary(all_doc_list)
# for i in dictionary:
#     print(dictionary[i])


# 2.2 制作语料库
# 历史文档的二元组向量转换
corpus = [dictionary.doc2bow(doc) for doc in all_doc_list]

# 测试文档的二元组向量转换
doc_test_vec = dictionary.doc2bow(doc_test_list)


# 3 相似度分析
# 3.1 使用TF-IDF模型对语料库建模
tfidf = models.TfidfModel(corpus)
# 获取测试文档中，每个词的TF-IDF值
tfidf[doc_test_vec]


# 3.2 对每个目标文档，分析测试文档的相似度
index = similarities.SparseMatrixSimilarity(tfidf[corpus], num_features=len(dictionary.keys()))
# 测试样本相似度与历史样本文档相似度
sim = index[tfidf[doc_test_vec]]
print(np.round(sim,3))
#提取相似度大于0.2的索引
arr_index= np.argwhere(sim >= 0.6)


arr_value=np.array(all_doc)[arr_index]
# print(len(arr_value))
# print(arr_value)
# print(all_doc[arr_index])


# 根3.3 据相似度排序
sort_result=sorted(enumerate(sim), key=lambda item: -item[1])
# print(sort_result)


