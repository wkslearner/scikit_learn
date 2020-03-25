import logging
import logging.config

# 读取日志配置文件内容
logging.config.fileConfig('./logging.conf')

# 创建一个日志器logger
logger = logging.getLogger('simpleExample')



# 日志输出
logger.debug('debug message')
logger.info('info message')
# logger.warn('warn message')
logger.error('error message')
logger.critical('critical message')





