# from datetime import datetime
import time
import os

curr = os.path.dirname(os.path.abspath(__file__))
# curr = os.path.dirname(__file__)
logfile = '/timelog.txt'


def logtime(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()

        result = func(*args, **kwargs)

        total_time = time.time() - start_time

        with open(curr + logfile, 'a') as log_t:
            log_t.write(
                f"{time.ctime()}\t{func.__name__}\t{total_time}\n")

        return result
    return wrapper


@logtime
def slow(a, b):
    time.sleep(1)
    return a + b


if __name__ == '__main__':
    slow(3, 4)
    print(curr)
