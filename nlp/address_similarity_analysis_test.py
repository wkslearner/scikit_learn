import  jieba
import pandas as pd
import numpy as np
from jieba import analyse
from gensim import corpora,models,similarities
import time


dataset=pd.read_excel('/Users/admin/Documents/data_analysis/fraud_model/data_source/fraud_data_v1.0.xlsx')
dataset['address']=dataset['address'].fillna('null')
dataset=dataset[dataset['address']!='null']
train_dataset=dataset[dataset['dt_created']<'2018-12-01']
test_dataset=dataset[dataset['dt_created']>='2018-12-01']
all_doc=train_dataset['address']
# print(train_dataset.shape[0])
# print(test_dataset.shape[0])
print('step one over')


#对历史数据提取关键词
all_doc_list = []
for doc in all_doc:
    doc=doc.replace(' ', '')
    doc_list = [word for word in analyse.extract_tags(doc)]
    # print(doc)
    # print(doc_list)
    all_doc_list.append(doc_list)


#比较数据关键词提取
test_doc_list=[]
for trand_no,doc_test in zip(test_dataset['trade_no'],test_dataset['address']):
    doc_test=doc_test.replace(' ', '')
    test_list = [word for word in analyse.extract_tags(doc_test)]
    # test_doc_list.append([trand_no,test_list,doc_test])
    test_doc_list.append(test_list)

print('step two over')


#获取词袋
dictionary = corpora.Dictionary(all_doc_list)
#转换为2维向量
corpus = [dictionary.doc2bow(doc) for doc in all_doc_list]

#使用TF-IDF模型对语料库建模
tfidf = models.TfidfModel(corpus)
print('step three over')

start_time=time.time()
test_corpus=[dictionary.doc2bow(doc) for doc in test_doc_list]

index = similarities.SparseMatrixSimilarity(tfidf[corpus], num_features=len(dictionary.keys()))
sim = index[tfidf[test_corpus]]
# arr_index= np.argwhere(sim >= 0.8)

all_data=np.array(['0'])
#重置索引（借用dataframe对数据进行批量处理，提升处理效率）
test_trade_list=test_dataset['trade_no'].reset_index(drop=True)
for i in range(len(sim)):
    #匹配相似度达到阈值的索引
    arr_index = np.argwhere(sim[i] >= 0.7)
    train_trade_no=np.array(train_dataset['trade_no'])[arr_index]
    if len(train_trade_no)>0:
        #输出相似度数据列表
        test_trade_no = test_trade_list[i]
        merge_trade_no=np.vstack((test_trade_no,train_trade_no))
        all_data=np.vstack((all_data,merge_trade_no))

print(time.time() - start_time)



'''进行单个文本转换方式（效率较低）'''
# start_time=time.time()
# test_vec_list=[]
# all_data=np.array(['',''])
# train_target=np.array(train_dataset[['trade_no','address']])
# for item in test_doc_list:
#
#     trand_no=item[0]
#     test_vec=dictionary.doc2bow(item[1])
#     test_address=item[2]
#
#     # 3.2 对每个目标文档，分析测试文档的相似度
#     index = similarities.SparseMatrixSimilarity(tfidf[corpus], num_features=len(dictionary.keys()))
#
#     sim = index[tfidf[test_vec]]
#     # print(np.round(sim,3))
#
#     #提取相似度大于0.8的索引
#     arr_index= np.argwhere(sim >= 0.8)
#
#     # #当前trade_no 所在索引
#     # this_index=np.argwhere(np.array(test_doc_list)[:,0]==trand_no)
#     # #合并索引
#     # merge_index=np.vstack((this_index,arr_index))
#
#     this_arr=[trand_no,test_address]
#     arr_value=np.array(all_doc)[arr_index]
#
#     # if  this_index not in all_index and len(arr_value)>0:
#     #     all_index=np.vstack((all_index,merge_index))
#
#     print(time.time() - start_time)
#     index_list=[]
#     if len(arr_value)>0:
#         print(this_arr)
#         print(train_target[arr_index][:,0])
#         #使用argwhere生成的index做索引是数组会多一层，需要手动去掉
#         merge_arr=np.vstack((this_arr,train_target[arr_index][:,0]))
#         all_data=np.vstack((all_data,merge_arr))



repeat_df=pd.DataFrame(all_data,columns=['trade_no'])
end_df=pd.merge(repeat_df,dataset,on='trade_no',how='left')

end_df.to_excel('/Users/admin/Documents/data_analysis/fraud_model/data_source/fraud_similar.xlsx')
