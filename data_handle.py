#!/usr/bin/python
# encoding=utf-8

import pandas as pd
import random

user_info=pd.read_excel('/Users/andpay/Documents/job/mode/user-info.xlsx')
#去除没有人行报告的数据
#user_info=user_info[user_info['applyid'].astype(int)>151062]

m2_df=pd.read_csv('/Users/andpay/Documents/job/mode/M2_list.csv')


def party_list():
    #根据partyid 随机抽取applyid
    result_list=[]
    for partyid in m2_df['PARTYID'].unique():
        if partyid  in user_info['partyid'].unique():
            applyid_list=user_info[user_info['partyid']==partyid]['applyid']
            applyid_list=list(applyid_list).sort(1)
            print(applyid_list)
            applyid=applyid_list[0]
            result_list.append(applyid)

    return result_list


ls=party_list()
df=pd.DataFrame(ls,columns=['applyid'])

end_user_df=user_info[user_info['applyid'].isin(df['applyid'])]

excel_writer=pd.ExcelWriter('/Users/andpay/Documents/job/mode/user-info-sample.xlsx',engine='xlsxwriter')
end_user_df.to_excel(excel_writer,index=False)
excel_writer.save()


