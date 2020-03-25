
'''
log 基础
官方文档：https://docs.python.org/zh-cn/3/howto/logging.html
https://cuiqingcai.com/6080.html
'''

import logging

# # log 全局配置
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# # 声明log 对象 logger
# logger = logging.getLogger(__name__)
#
# logger.info('This is a log info')
# logger.debug('Debugging')
# logger.warning('Warning exists')
# logger.info('Finish')


'''
log 基本操作
在这里我们首先引入了 logging 模块，然后进行了一下基本的配置，这里通过 basicConfig 配置了 level 信息和 format 信息，
这里 level 配置为 INFO 信息，即只输出 INFO 级别的信息，另外这里指定了 format 格式的字符串，
包括 asctime、name、levelname、message 四个内容，分别代表运行时间、模块名称、日志级别、日志内容，
这样输出内容便是这四者组合而成的内容了，这就是 logging 的全局配置。

接下来声明了一个 Logger 对象，它就是日志输出的主类，调用对象的 info() 方法就可以输出 INFO 级别的日志信息，
调用 debug() 方法就可以输出 DEBUG 级别的日志信息，非常方便。在初始化的时候我们传入了模块的名称，
这里直接使用 __name__ 来代替了，就是模块的名称，如果直接运行这个脚本的话就是 __main__，
如果是 import 的模块的话就是被引入模块的名称，这个变量在不同的模块中的名字是不同的，所以一般使用 __name__ 来表示就好了，
再接下来输出了四条日志信息，其中有两条 INFO、一条 WARNING、一条 DEBUG 信息
'''


'''
logging basicConfig参数：

filename：即日志输出的文件名，如果指定了这个信息之后，实际上会启用 FileHandler，而不再是 StreamHandler，这样日志信息便会输出到文件中了。
filemode：这个是指定日志文件的写入方式，有两种形式，一种是 w，一种是 a，分别代表清除后写入和追加写入。
format：指定日志信息的输出格式，即上文示例所示的参数，详细参数可以参考：docs.python.org/3/library/l…，部分参数如下所示：
%(levelno)s：打印日志级别的数值。
%(levelname)s：打印日志级别的名称。
%(pathname)s：打印当前执行程序的路径，其实就是sys.argv[0]。
%(filename)s：打印当前执行程序名。
%(funcName)s：打印日志的当前函数。
%(lineno)d：打印日志的当前行号。
%(asctime)s：打印日志的时间。
%(thread)d：打印线程ID。
%(threadName)s：打印线程名称。
%(process)d：打印进程ID。
%(processName)s：打印线程名称。
%(module)s：打印模块名称。
%(message)s：打印日志信息。
datefmt：指定时间的输出格式。
style：如果 format 参数指定了，这个参数就可以指定格式化时的占位符风格，如 %、{、$ 等。
level：指定日志输出的类别，程序会输出大于等于此级别的信息。
级别从高到低 分别为
CRITICAL	50
FATAL	50
ERROR	40
WARNING	30
WARN	30
INFO	20
DEBUG	10
NOTSET	0
stream：在没有指定 filename 的时候会默认使用 StreamHandler，这时 stream 可以指定初始化的文件流。
handlers：可以指定日志处理时所使用的 Handlers，必须是可迭代的。
'''

# import logging
#
# logging.basicConfig(level=logging.DEBUG,
#                     filename='output.log',
#                     datefmt='%Y/%m/%d %H:%M:%S',
#                     format='%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(module)s - %(message)s')
# logger = logging.getLogger(__name__)
#
# logger.info('This is a log info')
# logger.debug('Debugging')
# logger.warning('Warning exists')
# logger.info('Finish')



'''
Handler 的用法

先声明了一个 Logger 对象，然后指定了其对应的 Handler 为 FileHandler 对象，
然后 Handler 对象还单独指定了 Formatter 对象单独配置输出格式，最后给 Logger 对象添加对应的 Handler。

StreamHandler：logging.StreamHandler；日志输出到流，可以是 sys.stderr，sys.stdout 或者文件。
FileHandler：logging.FileHandler；日志输出到文件。
BaseRotatingHandler：logging.handlers.BaseRotatingHandler；基本的日志回滚方式。
RotatingHandler：logging.handlers.RotatingHandler；日志回滚方式，支持日志文件最大数量和日志文件回滚。
TimeRotatingHandler：logging.handlers.TimeRotatingHandler；日志回滚方式，在一定时间区域内回滚日志文件。
SocketHandler：logging.handlers.SocketHandler；远程输出日志到TCP/IP sockets。
DatagramHandler：logging.handlers.DatagramHandler；远程输出日志到UDP sockets。
SMTPHandler：logging.handlers.SMTPHandler；远程输出日志到邮件地址。
SysLogHandler：logging.handlers.SysLogHandler；日志输出到syslog。
NTEventLogHandler：logging.handlers.NTEventLogHandler；远程输出日志到Windows NT/2000/XP的事件日志。
MemoryHandler：logging.handlers.MemoryHandler；日志输出到内存中的指定buffer。
HTTPHandler：logging.handlers.HTTPHandler；通过”GET”或者”POST”远程输出到HTTP服务器。
'''

# # 报错日志
# import logging
#
# logger = logging.getLogger(__name__)
# logger.setLevel(level=logging.DEBUG)
#
# # Formatter
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#
# # FileHandler 日志输出到文件
# file_handler = logging.FileHandler('result.log')
# file_handler.setFormatter(formatter)
# logger.addHandler(file_handler)
#
# # StreamHandler 日志在界面打印
# stream_handler = logging.StreamHandler()
# stream_handler.setFormatter(formatter)
# logger.addHandler(stream_handler)
#
# # Log
# logger.info('Start')
# logger.warning('Something maybe fail.')
#
# try:
#     result = 10 / 0
# except Exception:
#     # 将错误提示打印到日志 exc_info 参数进行设定
#     logger.error('Faild to get result', exc_info=True)
# logger.info('Finished')




import logging

from log_use import log_form


logger = logging.getLogger('main')
logger.setLevel(level=logging.DEBUG)

# Handler
handler = logging.FileHandler('result.log')
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


logger.info('Main Info')
logger.debug('Main Debug')
logger.error('Main Error')
log_form.run()


