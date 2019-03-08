


import jieba
from gensim import corpora,models,similarities


all_doc=['北京市朝阳区建国门外大街','北京市建国门外大街5号院','北京市朝阳区外大街5号院']

# 1 分词
# 1.1 历史比较文档的分词
all_doc_list = []
for doc in all_doc:
    doc_list = [word for word in jieba.cut_for_search(doc)]
    print(doc_list)
    # doc_list = [word for word in jieba.cut(doc)]
    all_doc_list.append(doc_list)

# 1.2 测试文档的分词
doc_test="北京市朝阳区建国门外大街5号院"
doc_test_list = [word for word in jieba.cut_for_search(doc_test)]
print(doc_test_list)
# doc_test_list = [word for word in jieba.cut(doc_test)]


# 2 制作语料库
# 2.1 获取词袋
dictionary = corpora.Dictionary(all_doc_list)
print(dictionary.__class__)
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
sim = index[tfidf[doc_test_vec]]
print(sim)


# 根3.3 据相似度排序
sort_result=sorted(enumerate(sim), key=lambda item: -item[1])
print(sort_result)



