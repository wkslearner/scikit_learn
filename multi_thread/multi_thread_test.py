'''
使用多线程优化执行效率
'''

from random import randint
from threading import Thread
from time import time, sleep

'''单线程处理方式'''
def download_task(filename):
    print('开始下载%s...' % filename)
    time_to_download = 6
    #randint(5, 10)
    sleep(time_to_download)
    print('%s下载完成! 耗费了%d秒' % (filename, time_to_download))


def main_one():
    start = time()
    download_task('Python从入门到住院.pdf')
    download_task('Peking Hot.avi')
    end = time()
    print('总共耗费了%.2f秒.' % (end - start))


'''多线程处理方式'''
def download(filename):
    print('开始下载%s...' % filename)
    time_to_download = 6
        #randint(5, 10)
    sleep(time_to_download)
    print('%s下载完成! 耗费了%d秒' % (filename, time_to_download))


def main_two():
    start = time()
    t1 = Thread(target=download, args=('Python从入门到住院.pdf',))
    t1.start()  # 进程开始执行
    t2 = Thread(target=download, args=('Peking Hot.avi',))
    t2.start()
    t1.join() #等待进程执行结束
    t2.join()
    end = time()
    print('总共耗费了%.3f秒' % (end - start))




if __name__ == '__main__':
    main_one()
    main_two()