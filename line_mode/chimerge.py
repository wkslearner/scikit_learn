#!/usr/bin/python
# encoding=utf-8

from time import ctime
from sklearn.datasets  import  load_iris
import pandas as pd


'''
user_info=pd.read_excel('/Users/andpay/Documents/job/mode/random_applyid_data.xlsx')
m2_df=pd.read_csv('/Users/andpay/Documents/job/mode/M2_list.csv')

end_user_info=pd.merge(user_info,m2_df,left_on='partyid',right_on='PARTYID')
end_user_info=end_user_info.drop(['partyid','applyid','phone','PARTYID','city'],axis=1)
end_user_info.loc[end_user_info['CATEGROY']=='NM','CATEGROY']=0
end_user_info.loc[end_user_info['CATEGROY']=='M2','CATEGROY']=1
end_user_info=end_user_info[['CATEGROY','age','cardCount']]
#end_user_info=end_user_info.fillna(0)


iris=load_iris()
data=iris.data
category_data=iris.target
feature=iris.feature_names
targetname=iris.target_names
#print(iris)

ls=[]
for i in range(len(data)):
    ls.append(list(data[i]))

iris_df=pd.DataFrame(ls,columns=feature)
iris_df['category']=category_data
'''


'''重塑数据格式，把类似[4.5, 0.0, 1.0]转换为(4.5, [1.0, 0, 0])以便计算，
   表示在4.5区间内，类别为0的数量是1'''
def build(log_cnt):
    log_dict = {}
    for record in log_cnt:
        if record[0] not in log_dict.keys():
            log_dict[record[0]] = [0,0,0]
        if record[1] == 0:
            log_dict[record[0]][0] = record[2]
        elif record[1] == 1:
            log_dict[record[0]][1] = record[2]
        elif record[1] == 2:
            log_dict[record[0]][2] = record[2]
        else:
            raise TypeError('Data Exception')
    #排序之后，字典类型数据自动转化为list
    log_truple = sorted(log_dict.items())

    return log_truple



'''区间合并'''
def combine(a,b):
    '''''  a=('4.4', [3, 1, 0]), b=('4.5', [1, 0, 2]) 
           combine(a,b)=('4.4', [4, 1, 2])  '''
    c = a[:]
    for i in range(len(a[1])):
        c[1][i] += b[1][i]
    return c


def chi2(A):
    '''计算两个区间的卡方值'''
    m = len(A)
    k = len(A[0])
    R = []
    '''第i个区间的实例数'''
    for i in range(m):
        sum = 0
        for j in range(k):
            sum += A[i][j]
        R.append(sum)
    C = []
    '''第j个类的实例数'''
    for j in range(k):
        sum = 0
        for i in range(m):
            sum+= A[i][j]
        C.append(sum)
    N = 0
    '''总的实例数'''
    for ele in C:
        N +=ele
    res = 0.0
    for i in range(m):
        for j in range(k):
            #第i区间，第j类的期望值(频数)
            Eij = 1.0*R[i] *C[j]/N
            if Eij!=0:
                res = 1.0*res + 1.0*(A[i][j] - Eij)**2/Eij
    return res


'''ChiMerge 算法'''
'''下面的程序可以看出，合并一个区间之后相邻区间的卡方值进行了重新计算，而原作者论文中是计算一次后根据大小直接进行合并的
下面在合并时候只是根据相邻最小的卡方值进行合并的，这个在实际操作中还是比较好的
'''
def ChiMerge(log_tuple,max_interval):
    num_interval = len(log_tuple)
    while num_interval>max_interval:
        num_pair = num_interval -1
        chi_values = []
        ''' 计算相邻区间的卡方值'''
        for i in range(num_pair):
            arr = [log_tuple[i][1],log_tuple[i+1][1]]
            chi_values.append(chi2(arr))
        min_chi = min(chi_values)
        '''根据卡方结果对数据进行合并'''
        for i in range(num_pair - 1,-1,-1):
            if chi_values[i] == min_chi:
                log_tuple[i] = combine(log_tuple[i],log_tuple[i+1])
                log_tuple[i+1] = 'Merged'
        while 'Merged' in log_tuple:
            log_tuple.remove('Merged')
        num_interval = len(log_tuple)
    split_points = [record[0] for record in log_tuple]
    return split_points


'''获取分段节点'''
def discrete(dataframe,var,target_var,max_interval):
    dataframe['assist_col']=1


    '''统计每个区间和类别下的数据个数(空值会被自动过滤掉)'''
    data_agg = dataframe.groupby([dataframe[var],dataframe[target_var]]).agg({'assist_col':'count'}).reset_index()

    area_list = []
    for i in range(len(data_agg.index)):
        area_list.append(list(data_agg.ix[i, :]))

    '''数据转换'''
    log_tuple = build(area_list)

    '''计算分段节点'''
    split_points = ChiMerge(log_tuple,max_interval)


    return split_points


