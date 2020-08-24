
''''''

'''
threading 库可以在单独的线程中执行任何的在 Python 中可以调用的对象。
你可以创建一个 Thread 对象并将你要执行的对象以 target 参数的形式提供给该对象。
'''

import time
def countdown(n):
    while n > 0:
        print('T-minus', n)
        n -= 1
        time.sleep(5)

# Create and launch a thread
from threading import Thread
# 把要执行的函数传给target 参数
t = Thread(target=countdown, args=(10,))

# 查看线程运行状态
if t.is_alive():
    print('Still running')
else:
    print('Completed')


# 加入一个线程
t.join()

# 运行线程
t.start()






