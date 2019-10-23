import logging
logging.debug('This message should go to the log file')
logging.info('So should this')
logging.warning('And this, too')


def square(x):
    return x * x


if __name__ == '__main__':
    square(10)
