
import numpy as np
import decimal
from sklearn.datasets import load_iris
import pandas as pd


# #定义‘符号’变量，也称为占位符
# a = tf.placeholder("float")
# b = tf.placeholder("float")
#
# y = tf.multiply(a, b) #构造一个op节点
# z = tf.pow(a,b)       #进行幂次方
# sess = tf.Session()   #建立会话
#
# #运行会话，输入数据，并计算节点，同时打印结果
# print (sess.run(y, feed_dict={a: 3, b: 3}))
# print (sess.run(z,feed_dict={a:[2,3],b:[3,2]}))
#
# # 任务完成, 关闭会话.
# sess.close()


iris=load_iris()
data=iris.data
data=np.array(data)


# result=np.array([0,0,0,0])
# for i in range(data.shape[0]-1):
#     #print(data[i])
#     result=np.vstack((result,data[i]))
#     #result=np.concatenate((result,[data[i+1]]),axis=0)
#
# print(result)
#
# a = np.array([[1, 2], [3, 4]])
# b = np.array([[5, 6]])
# print(np.concatenate((a, b), axis=0))

# test_dict=[{'key':1,'value':2},{'key':np.nan,'value':2},{'key':3,'value':3}]
# df=pd.DataFrame.from_dict(test_dict)
#
# # print(df[df['key'].apply(lambda x:x if x)]['key'].astype(float))
# df=df.astype(float)
# df=df.drop(['key'],axis=1)
# print(df)

# 风控数据合并处理
# data_source=pd.read_excel('/Users/admin/Documents/data_analysis/model/data_dource/risk_data_source/risk_merge_data_v2.xlsx','data_source')
# add_data=pd.read_excel('/Users/admin/Documents/data_analysis/model/data_dource/risk_data_source/risk_merge_data_v2.xlsx','add_data')
#
# result_data=pd.merge(data_source,add_data,on='trade_no',how='left')
# print(result_data.shape[0])
#
# result_data.to_excel('/Users/admin/Documents/data_analysis/model/data_dource/risk_data_source/risk_merge_data_v2.0.xlsx')


fraud_data=pd.read_excel('/Users/admin/Documents/data_analysis/fraud_model/data_source/fraud_data_v1.0.xlsx')
longla_data=pd.read_excel('/Users/admin/Documents/data_analysis/fraud_model/data_source/longla_data_v1.xlsx')

merge_data=pd.merge(fraud_data,longla_data,on='trade_no',how='left')

merge_data.to_excel('/Users/admin/Documents/data_analysis/fraud_model/data_source/fraud_merge_data.xlsx')













