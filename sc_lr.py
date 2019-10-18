# Make sure that you have all these libaries available to run the code successfully

import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import urllib.request, json
import os
import numpy as np
import tensorflow as tf # This code has been tested with TensorFlow 1.6
from sklearn.preprocessing import MinMaxScaler
import time, functools
from functools import wraps


#装饰器
def metric(fn):
    @wraps(fn)
    def derator(*args, **kwargs):
        print('%s executed in %s ms' % (fn.__name__, 10.24))
        return fn
    return derator



# 测试
@metric
def fast(x, y):
    time.sleep(0.0012)
    return x + y;


@metric
def slow(x, y, z):
    time.sleep(0.1234)
    return x * y * z;


f = fast(11, 24)
s = slow(11, 22, 33)
if f != 33:
    print('测试失败!')
elif s != 7986:
    print('测试失败!')



