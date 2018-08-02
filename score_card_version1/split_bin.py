
import pandas as pd
from score_card_version1.preprocessing import monotone

'''变量分类'''
def category_var(dataframe):
    '''
    :param dataframe: 目标数据框
    :return: 分类变量列表和连续变量列表
    '''
    cat_list=[]
    con_list=[]
    column=dataframe.columns
    for var in column:
        value_list=dataframe[dataframe[var].notnull()][var].unique()
        if len(value_list)<=5:
            cat_list.append(var)
        else:
            con_list.append(var)

    return cat_list,con_list


'''分类变量处理'''
def disper_split(dataframe,var_list):
    '''
    :param dataframe: 目标数据框
    :param var_list: 分类变量列表
    :return: 变量与数值映射字典及分类处理后的新数据框
    '''
    split_point_cat={}
    split_cat_list = []
    for var in var_list:
        split_cat_list.append(var + '_cat')
        mid_dict={}
        if dataframe[dataframe[var].isnull()].shape[0] > 0:
            sort_value = sorted(list(dataframe[dataframe[var].notnull()][var].unique()))
            num = len(sort_value)
            for i in range(num):
                dataframe.loc[(dataframe[var] == sort_value[i]), var + '_cat'] = i
                mid_dict[i]=sort_value[i]

            dataframe.loc[dataframe[var].isnull(), var + '_cat'] = -1
            mid_dict[-1]='None'
            split_point_cat[var+'_cat']=mid_dict

        else:
            sort_value = sorted(list(dataframe[dataframe[var].notnull()][var].unique()))
            num = len(sort_value)
            for i in range(num):
                dataframe.loc[(dataframe[var] == sort_value[i]), var + '_cat'] = i
                mid_dict[i] = sort_value[i]

            split_point_cat[var + '_cat'] = mid_dict

    return split_point_cat,dataframe[split_cat_list]


'''连续变量进行等频分箱后，卡方分箱'''
def chi_equalwide(dataframe,var_list,target_var,mont=True,numOfSplit=200,max_interval=5,special_list=[]):
    '''
    :param dataframe: 目标数据框
    :param var_list: 需要处理的变量列表
    :param target_var: 模型因变量，需要先剔除
    :param mont: 是否要保证变量的单调性，可选True 或 False
    :param numOfSplit: 等频分箱的分箱数量，默认分为200箱
    :param max_interval: 需要最终的箱数，默认单调的情况下，箱数可能少于设置值
    :param special_list: 特殊变量列表，其中变量不要求默认单调
    :return: 变量与数值映射字典及分类处理后的新数据框
    '''
    split_point_freq = {}
    freq_var_list = []
    for var in var_list:
        freq_var_list.append(var + '_freq')
        split_point = basic_splitbin(dataframe, var, numOfSplit=numOfSplit)  # 先对数据进行等频分箱，分为100箱
        split_point_freq[var + '_freq'] = split_point
        num = len(split_point)

        for i in range(num - 1):
            before = split_point[i]
            after = split_point[i + 1]
            # 分箱后新设变量并重新赋值
            dataframe.loc[(dataframe[var] >= before) & (dataframe[var] < after), var + '_freq'] = i
            dataframe.loc[dataframe[var] >= split_point[num - 1], var + '_freq'] = i + 1  # 最后一个分箱单独处理

    '''构造特殊变量'''
    new_special_list = []
    for var in special_list:
        new_special_list.append(var + '_freq')

    '''等频变量卡方分箱'''
    split_point_chi={}
    chi_var_list=[]
    monotone_list=[]
    for var in freq_var_list:
        '''首次对数据分为5箱'''
        chi_var_list.append(var+'_bin')
        max_int=max_interval      #每个变量分箱时初始化max_int
        split_point=ChiMerge(dataframe,var,target_var,max_interval=max_int)  #卡方分箱，返回分箱分割点
        freq_value=split_point_freq[var] #引入等频分箱的数据分割点列表

        num = len(split_point)
        mid_dict = {}

        for i in range(num-1):
            before=split_point[i]
            after=split_point[i+1]
            min_value=freq_value[int(before)]
            max_value=freq_value[int(after)]
            bin_name=str(min_value) + '-' + str(max_value)
            #分箱后新设变量并重新赋值
            dataframe.loc[(dataframe[var] >= before) & (dataframe[var] < after), var+'_bin'] = i
            mid_dict[i] =bin_name  #保存分箱范围到字典

        #最后一个分箱处理
        dataframe.loc[dataframe[var] >= split_point[num - 1], var+'_bin'] = i + 1
        specil_value=freq_value[int(split_point[num - 1])]
        mid_dict[i+1]=str(specil_value)+'-'+'inf'

        '''单调性检验,并做重新分箱处理'''
        mot = monotone(dataframe, var + '_bin', target_var)  # 单调性检验


        if mont==True:
            '''如果存在其他业务上可理解的非单调性变量（如年龄），需要申明special_list'''
            if var not in new_special_list:
                while (not mot):
                    max_int -= 1
                    split_point = ChiMerge(dataframe, var, target_var, max_interval=max_int)
                    num = len(split_point)
                    mid_dict = {}

                    for i in range(num - 1):
                        before = split_point[i]
                        after = split_point[i + 1]
                        min_value = freq_value[int(before)]
                        max_value = freq_value[int(after)]
                        bin_name = str(min_value) + '-' + str(max_value)
                        # 分箱后新设变量并重新赋值
                        dataframe.loc[(dataframe[var] >= before) & (dataframe[var] < after), var + '_bin'] = i
                        mid_dict[i] = bin_name  #保存分箱范围到字典

                    dataframe.loc[dataframe[var] >= split_point[num - 1], var + '_bin'] = i + 1
                    specil_value = freq_value[int(split_point[num - 1])]
                    mid_dict[i + 1] = str(specil_value) + '-' + 'inf'

                    mot = monotone(dataframe, var+'_bin', target_var)


        monotone_list.append([var+'_bin',mot])
        dataframe.loc[dataframe[var].isnull(),var+'_bin']=-1  #空值分箱单独处理
        mid_dict[-1]=['None']
        split_point_chi[var+'_bin']=mid_dict

    return split_point_chi,dataframe[chi_var_list]


'''等频或等宽分箱法'''
def basic_splitbin(df,var,numOfSplit = 5, method = 'equal_freq'):
    '''
    :param df: 数据集
    :param var: 需要分箱的变量。仅限数值型。
    :param numOfSplit: 需要分箱个数，默认是5
    :param method: 分箱方法，'equal freq'：，默认是等频，否则是等距
    :return:分割点列表
    '''
    if method == 'equal_freq':
        notnull_df = df.loc[~df[var].isnull()]
        N = notnull_df.shape[0]
        n = int(N / numOfSplit)  #每箱数据条数
        splitPointIndex = [i * n for i in range(1, numOfSplit)]
        rawValues = sorted(list(notnull_df[var]))   #对目标变量进行排序
        maxvalue=max(notnull_df[var])
        minvalue=min(notnull_df[var])
        splitPoint = [rawValues[i] for i in splitPointIndex]
        splitPoint = sorted(list(set(splitPoint)))
        if splitPoint[0]>minvalue:
            splitPoint.insert(0,minvalue) #向分割点第一个位置插入最小值
        return splitPoint
    elif method=='equal_wide':
        var_max, var_min = max(df[var]), min(df[var])
        interval_len = (var_max - var_min)*1.0/numOfSplit
        splitPoint = [var_min + i*interval_len for i in range(1,numOfSplit)]
        return splitPoint
    else:
        print('the method do not exist')


'''-------------------以下是卡方分箱法————————————————————————————'''

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
def ChiMerge(dataframe,var,target_var,max_interval=5):
    dataframe['assist_col'] = 1

    '''统计每个区间和类别下的数据个数(空值会被自动过滤掉)'''
    data_agg = dataframe.groupby([dataframe[var], dataframe[target_var]]).agg({'assist_col': 'count'}).reset_index()

    area_list = []
    for i in range(len(data_agg.index)):
        area_list.append(list(data_agg.ix[i, :]))

    '''数据转换'''
    log_tuple = build(area_list)

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


