'''将打印内容 输出到log文件'''

import sys
import logging

logging.basicConfig(level=logging.DEBUG,
                    filename='print_message.log',
                    datefmt='%Y/%m/%d %H:%M:%S',
                    format='%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(module)s - %(message)s')
logger = logging.getLogger(__name__)


# make a copy of original stdout route
stdout_backup = sys.stdout
# define the log file that receives your log info
log_file = open("message.log", "w")
# redirect print output to log file
sys.stdout = log_file

# print ("Now all print info will be written to message.log")

# any command line that you will execute

...

log_file.close()
# restore the output to initial pattern
sys.stdout = stdout_backup

def log_readline():
    file=open('message.log','r')
    for line in file.readlines():
        logger.info(line)

log_readline()

print ("Now this will be presented on screen")
print ('this is a wrong word division by zero')




