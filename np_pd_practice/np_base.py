
import pandas as pd
import numpy as np


df=pd.DataFrame({'a':['xxx','aaa','bbb','ccc','ddd','eee'],'b':[0,2,1,3,5,1],'c':[1,1,2,3,4,1]})
print(df['a'][2:5].reset_index(drop=True)[0])

#把数据框转换成数组
arr_df=np.array(df[['a','b','c']])

# #求指定条件索引
# condition_index_one=np.argwhere(arr_df[:,1]>2)
# condition_index_two=np.argwhere(arr_df[:,1]==1)
#
# #指定条件索引合并vstack进行列方向合并，hstack进行行方向合并
# merge_index=np.vstack((condition_index_one,condition_index_two))

# print(df.groupby(['a'],as_index= False).agg({'b':'sum'}).rename(columns={'b':'sum_count'}))
# print(arr_df[arr_df[:,1]!=2])





# N = 50  # number of points to solve at
#
# # K = X.shape[1]
# # coeff = np.zeros((N, K))  # Holds the coefficients
#
# alphas = 1 / np.logspace(-0.5, 2, N)
#
# print(alphas)

print(1/np.logspace(-0.5,2,10))


