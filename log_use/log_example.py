import logging

def runs():
    a=10/0
    return a

def main():
    logging.basicConfig(filename='myapp.log', level=logging.INFO)
    logging.info('Started')
    try:
        runs()
    except:
        logging.error('error log',exc_info=True)
    logging.info('Finished')

if __name__ == '__main__':
    main()