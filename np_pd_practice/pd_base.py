

import pandas as pd
import numpy as np
import json

df=pd.DataFrame({'a':['bb,bb','a,aa','bbb','aaa','ddd','eee'],'b':[0,2,np.nan,3,5,1],'c':[1,1,2,3,4,1],
                 'd':[3,5,6,7,1,2]})

'''pandas统计分析'''
#结合使用groupby+agg+reanme函数
# stat_analysis=df.groupby(by='a',as_index=False).agg({'b':'sum'}).rename(columns={'b':'b_sum'})

#分类统计,并排序
# print(pd.value_counts(df['b'],sort = True))
#
# #空值检测,是否存在空值
# print(df.isnull().values.any())
#
# print(df.groupby('a'))

#数据偏移
# df['d']=df['b'].shift(1)

#对数据进行组内排序后，并进行赋值（实现类似oracle中的分析函数功能）
# df['sort_num']=(df['b'].groupby(by=df['a'])).rank(method='dense')
# print(df.sort_values(by='a'))


#使用pandas 实现 count + distinct 的功能
# agg_result=df.groupby(by=['a']).b.nunique()


'''多层索引操作'''
#索引排序，
# df=df.sort_index(axis=1,ascending=[0,1],level=[1,0]).reset_index()

#索引层级交换
# df=df.swaplevel(0,axis=1)

#行列交换
# df=df.swapaxes(0,1)

#指定索引序列
# df=df.reindex(columns=['d','c'],level=0)



'''pandas-apply函数应用'''
# def filter_func(df):
#     if df['b']>3 and df['c']>2:
#         return True
#     elif df['b']>2 and df['c']>2:
#         return True
#     else :
#         return  False
#
# df['filter_status']=df.apply(lambda x:filter_func(x),axis=1)

import datetime as dt


print(list(df['a'].apply(lambda x:x.split(','))))
# df['word']=df.apply(lambda x: 1 if 'a' in x.a+str(x.b) else 0,axis=1)
# print(df)












