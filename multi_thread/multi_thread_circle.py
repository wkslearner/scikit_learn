''''''
'''
循环并发多线程
'''

#!/usr/bin/env python
# -*- coding: utf8 -*-

import sys
import time
import string
import threading
import datetime
import numpy as np
import time

st=time.time()

#  定义多线程循环调用函数
def MainRange(start, stop):  # 提供列表index起始位置参数
    for i in range(start, stop):
        time.sleep(0.1)

lens=int(1003/4)+1

threads = []
for i in range(4):
    t=threading.Thread(target=MainRange, args=(i*lens,(i+1)*lens))
    threads.append(t)


for t in threads:
    t.setDaemon(True)
    t.start()

t.join()
print ("multi_thread",time.time()-st)

sts=time.time()
MainRange(0,1003)
print("single_thread",time.time()-sts)



