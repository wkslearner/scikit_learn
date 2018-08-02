#!/usr/bin/python
# encoding=utf-8

import logging
import time
from time import ctime
from sklearn.datasets  import  load_iris


'''
log_name='/Users/andpay/Documents/python-file/logger_test.txt'
logging.basicConfig(filename=log_name,level=logging.DEBUG)
logging.debug('this is a loger txt')

logger=logging.getLogger()
handler=logging.FileHandler("Log_test.txt")
logger.addHandler(handler)
logger.setLevel(logging.NOTSET)
logger.error("This is an error message")
logger.info("This is an info message")
logger.critical("This is a critical message")
'''

iris=load_iris()
data=iris.data
category_data=iris.target
feature=iris.feature_names
targetname=iris.target_names
#print(iris)

ls=[]
for i in range(len(data)):
    turn=list(data[i])
    turn.append(category_data[i])
    ls.append(turn)


''' 读取数据'''
def read(file):
    Instances = []
    fp = open(file,'r')
    for line in fp:
        line = line.strip('\n')
        if line!='':
            Instances.append(line.split(','))
    fp.close()
    return Instances

''' 将第i个特征和类标签组合起来
 如:[[0.2,'Iris-setosa'],[0.2,'Iris-setosa'],...]'''

def split(Instances,i):
    log = []
    for line in Instances:
        log.append([line[i],line[4]])
    return log


''' 统计每个属性值所具有的实例数量
 [['4.3', 'Iris-setosa', 1], ['4.4', 'Iris-setosa', 3],...]'''
def count(log):
    log_cnt = []
    # 以第0列进行排序的 升序排序
    log.sort(key = lambda log:log[0])
    i = 0
    while(i<len(log)):
        cnt = log.count(log[i])
        record = log[i][:]
        record.append(cnt)
        log_cnt.append(record)
        i += cnt
    return log_cnt


''' log_cnt  是形如： ['4.4', 'Iris-setosa', 3] 的
 统计对于某个属性值，对于三个类所含有的数量量
 返回结果形如：{4.4:[0,1,3],...} 属性值为4.4的对于三个类的实例数量分别是：0、1、3 '''


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



def collect(Instances,i):
    log = split(Instances,i)
    log_cnt = count(log)
    log_tuple = build(log_cnt)


    return log_tuple


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
        for i in range(num_pair - 1,-1,-1):
            if chi_values[i] == min_chi:
                log_tuple[i] = combine(log_tuple[i],log_tuple[i+1])
                log_tuple[i+1] = 'Merged'
        while 'Merged' in log_tuple:
            log_tuple.remove('Merged')
        num_interval = len(log_tuple)
    split_points = [record[0] for record in log_tuple]
    return split_points


def discrete(path):
    Instances = path
    max_interval = 6
    num_log = 4
    for i in range(num_log):
        log_tuple = collect(Instances,i)
        split_points = ChiMerge(log_tuple,max_interval)
        print (split_points)

discrete(ls)

'''
if __name__=='__main__':
    print('Start: ' + ctime())
    discrete('iris.data')
    print('End: ' + ctime())
'''