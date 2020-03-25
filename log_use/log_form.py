'''
log 文件配置的3种形式
https://www.cnblogs.com/yyds/p/6885182.html
'''

import logging
import sys

'''
基于python代码的日志配置
'''

# 创建一个日志器logger并设置其日志级别为DEBUG
logger = logging.getLogger('simple_logger')
logger.setLevel(logging.DEBUG)

# 创建一个流处理器handler并设置其日志级别为DEBUG
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)

# 创建一个格式器formatter并将其添加到处理器handler
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)

# 为日志器logger添加上面创建的处理器handler
logger.addHandler(handler)

# 日志输出
logger.debug('debug message')
logger.info('info message')
logger.warn('warn message')
logger.error('error message')
logger.critical('critical message')



'''
基于文件生成的日志配置
'''
# 读取日志配置文件内容
logging.config.fileConfig('logging.conf')

# 创建一个日志器logger
logger = logging.getLogger('simpleExample')

# 日志输出
logger.debug('debug message')
logger.info('info message')
logger.warn('warn message')
logger.error('error message')
logger.critical('critical message')


'''
基于字典配置信息配置日志
'''
import logging
import logging.config
import yaml

with open('logging.yml', 'r') as f_conf:
    dict_conf = yaml.load(f_conf)
logging.config.dictConfig(dict_conf)

logger = logging.getLogger('simpleExample')
logger.debug('debug message')
logger.info('info message')
logger.warn('warn message')
logger.error('error message')
logger.critical('critical message')


