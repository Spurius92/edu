import time
import os

curr = os.path.dirname(os.path.abspath(__file__))
logfile = '/timelog.txt'


def once_per_minute(func):
    last_invoked = 0
    print('last:', last_invoked)
    def wrapper(*args, **kwargs):

        nonlocal last_invoked

        elapsed_time = time.time() - last_invoked
        print('elapsed time: ', elapsed_time)

        if elapsed_time < 60:
            raise RuntimeWarning('message')

        with open(curr + logfile, 'a') as log_t:
            log_t.write(
                f"{time.ctime()}\t{func.__name__}\n")

        last_invoked = time.time()
        print(last_invoked)

        return func(*args, **kwargs)

    return wrapper


@once_per_minute
def nonlocal_add(a, b):
    time.sleep(1)
    return a + b


if __name__ == '__main__':
    print(nonlocal_add(3, 4))
