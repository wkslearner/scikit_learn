#!/usr/bin/python
# encoding=utf-8

import pandas as pd
import numpy as np
import random


class RandomSample():

    def __init__(self,dataset,coefficient):
        self.dataset=dataset
        self.coefficient=coefficient


    '''随机欠采样'''
    def random_under_sample(self):

        len_data=len(self.dataset)
        sample_num=int(self.coefficient*len_data)
        result_data=random.sample(list(self.dataset),sample_num)

        return np.array(result_data)


    '''随机过采样'''
    def random_over_sample(self):

        len_data = len(self.dataset)
        result_num = int(len_data*self.coefficient)-len_data
        result_data=self.dataset

        #每次抽取的数量
        every_extract_num=2*int(self.coefficient)

        #每次随机抽取原来数组的二分之一的数据,最后一次抽取余下的数字
        for i  in range(every_extract_num):
            if i<every_extract_num-1:
                sample_num=int(result_num/(every_extract_num))
                mid_data = random.sample(list(self.dataset), sample_num)
                #合并数组
                result_data =np.vstack((result_data,mid_data))
            else:
                sample_num =result_num-int(result_num/(every_extract_num))*i
                mid_data = random.sample(list(self.dataset), sample_num)
                #合并数组
                result_data =np.vstack((result_data,mid_data))

        return result_data



if __name__=='__main__':
    data=np.array(np.random.randint(10,100,68))
    data=data.reshape([17,4])
    object=RandomSample(data,2.8)
    end=object.random_over_sample()
    print(end.shape)

