'''
对关联数据做唯一性处理
'''


import pandas as pd
import numpy as np
import math
import time


start_time=time.time()
dataset=pd.read_csv('/Users/admin/Documents/data_analysis/fraud_model/analysis_phone_book/phone_unique.csv')

dataset=dataset.fillna('None')
dataset=dataset.astype(str)



'''处理数据(对电话进行拆分处理)'''
data_list=[]
i=0
exist_identity=[]
identity_dict={}
for line in np.array(dataset):
    id_user=line[1];identity=line[2]
    sendphone = line[3].split(',');regphone = line[4].split(',');aliphone = line[5].split(',')
    all_phone = list(set(sendphone + regphone + aliphone))

    if 'None' in all_phone:
        all_phone.remove('None')

    if '' in all_phone:
        all_phone.remove('')

    if '0' in all_phone:
        all_phone.remove('0')

    if identity in exist_identity and identity!='None':
        uid=identity_dict[identity]
        for  phone in all_phone:
            data_list.append([uid,id_user,identity,phone])
    else:
        uid='uid'+str(i)
        exist_identity.append(identity) #添加判断列表

        #存储uid信息，方便后续查询
        identity_dict[identity]=uid
        for phone in all_phone:
            data_list.append([uid,id_user,identity,phone])
        i=i+1


result_df=pd.DataFrame(data_list,columns=['uid','id_user','identity','phone'])
# print(result_df.head())

backup_one=result_df.copy()
merge_data=pd.merge(result_df,backup_one[['uid','phone']],on='phone',how='left')

print(merge_data.shape[0])
merge_data.to_csv('/Users/admin/Documents/data_analysis/fraud_model/analysis_phone_book/phone_unique_merge.csv',index=False)
print(time.time()-start_time)









