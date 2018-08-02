import pandas as pd
import numpy as np
import information_woe as iw

user_info=pd.read_excel('/Users/andpay/Documents/job/mode/random_applyid_data.xlsx')

m2_df=pd.read_csv('/Users/andpay/Documents/job/mode/M2_list.csv')

end_user_info=pd.merge(user_info,m2_df,left_on='partyid',right_on='PARTYID')
end_user_info=end_user_info.drop(['partyid','applyid','phone','PARTYID','city'],axis=1)
end_user_info.loc[end_user_info['CATEGROY']=='NM','CATEGROY']=0
end_user_info.loc[end_user_info['CATEGROY']=='M2','CATEGROY']=1



def continuious_handle(dataframe,key,split_num):
    nomal_value=dataframe[key][dataframe[key].notnull()]
    data_set=list(nomal_value.astype(float))
    data_set.sort()
    min_value=min(data_set)
    max_value=max(data_set)


    length=round((max_value-min_value+1)/split_num,1)
    string = 'new' + '_' + key
    dataframe[string] = ''

    for i in range(split_num):
        split_point=min_value+i*length
        split_point_next=min_value+(i+1)*length
        split_string=str(split_point)+'-'+str(split_point_next)

        dataframe.loc[(dataframe[key]>=split_point)&(dataframe[key]<split_point_next),string]=split_string

    dataframe.loc[dataframe[key].isnull(),string]=np.nan
    #dataframe = dataframe.drop([key], axis=1)

    return dataframe

end_user_info=continuious_handle(end_user_info,'cardCount',5)




woe,information=iw.information_value(end_user_info,'new_cardCount','CATEGROY')

