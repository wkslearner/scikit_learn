import pandas as pd
import random
from sklearn import tree
import numpy as np
import information_woe as iw

user_info=pd.read_excel('/Users/andpay/Documents/job/mode/random_applyid_data.xlsx')
m2_df=pd.read_csv('/Users/andpay/Documents/job/mode/M2_list.csv')
#new_df=user_info.groupby(user_info['partyid']).agg({'applyid':'max'}).reset_index()
#new_df=user_info[user_info['applyid'].isin(new_df['applyid'])]

column=user_info.columns

use_list=[]
unuse_list=[]
for col in column:
    res=iw.check_nullvalue(user_info,col)
    if res=='unuseable':
        unuse_list.append(col)
    else:
        use_list.append(col)


print(use_list)
print(unuse_list)



'''
excel_writer=pd.ExcelWriter('/Users/andpay/Documents/job/mode/xx.xlsx',engine='xlsxwriter')
end_df.to_excel(excel_writer,index=False)
excel_writer.save()
'''