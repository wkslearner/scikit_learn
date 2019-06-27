
'''
收货地址关键词分析
'''


import jieba
from jieba import analyse
import pandas as pd
import time

string='江苏省无锡市 宜兴市阳羡西路213号'

# 用户设置停用词集合接口
def set_stop_words(stop_words_path):
    # 更新对象default_tfidf中的停用词集合
    analyse.tfidf.set_stop_words(stop_words_path)
    # 更新对象default_textrank中的停用词集合
    analyse.textrank.set_stop_words(stop_words_path)


address_df=pd.read_excel('/Users/admin/Documents/data_analysis/fraud_model/data_source/fraud_merge_data_result.xlsx','Sheet1')
address_df=address_df.fillna('null')
# address_df=address_df[0:1000]

start_time=time.time()

test_doc_list=[]
for trand_no,doc_test,user_cate in zip(address_df['trade_no'],address_df['address_x'],address_df['user_cate']):
    doc_test=doc_test.replace(' ', '')
    test_list = [word for word in analyse.extract_tags(doc_test)]
    # test_doc_list.append([trand_no,test_list,doc_test])
    for  item  in  test_list:
        test_doc_list.append([trand_no,item,user_cate])


keyword_df=pd.DataFrame(test_doc_list,columns=['trade_no','key_word','user_cate'])
print(keyword_df.head())
keyword_df=keyword_df.drop_duplicates()

static_keyword=keyword_df.groupby(['key_word','user_cate']).count().reset_index()
print(static_keyword.head())

sum_stat=static_keyword.groupby('key_word').agg({'trade_no':'sum'}).reset_index()
m2_stat=static_keyword[static_keyword['user_cate']=='M2'][['key_word','trade_no']]
merge_stat=pd.merge(sum_stat,m2_stat,on='key_word',how='left')
merge_stat['trade_no_y']=merge_stat['trade_no_y'].fillna(0)
merge_stat['m2_rate']=merge_stat['trade_no_y']/merge_stat['trade_no_x']
print(merge_stat.head())
print(time.time()-start_time)

merge_stat.to_excel('/Users/admin/Documents/data_analysis/fraud_model/data_source/fraud_keyword_analysis.xlsx')















