import pandas as pd
import time
import numpy as np

dataset=pd.read_csv('/Users/admin/Downloads/creditcard.csv')
arr_dataset=np.array(dataset)


start_time=time.time()
num_list=[0.1,0.2,0.3,0.4]
for value in num_list:
    data=dataset[dataset['V2']>value].shape[0]
    print(data)

print(time.time()-start_time)

start_time1=time.time()
num_list=[0.1,0.2,0.3,0.4]
for value in num_list:
    data=arr_dataset[np.where(arr_dataset[:,2]>value)].shape[0]
    print(data)

print(time.time()-start_time1)



'''
excel_writer=pd.ExcelWriter('/Users/andpay/Documents/job/mode/xx.xlsx',engine='xlsxwriter')
end_df.to_excel(excel_writer,index=False)
excel_writer.save()
'''