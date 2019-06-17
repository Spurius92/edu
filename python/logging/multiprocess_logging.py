import logging
import time
from multiprocessing import Process, Lock, Value
from multiprocessing import log_to_stderr, get_logger

from logtime import logtime


@logtime
def add_500_lock(total, lock):
    for i in range(100):
        time.sleep(0.01)
        lock.acquire()
        total.value += 5
        lock.release()


@logtime
def sub_500_lock(total, lock):
    for i in range(100):
        time.sleep(0.01)
        lock.acquire()
        total.value -= 5
        lock.release()


if __name__ == '__main__':

    total = Value('i', 500)
    lock = Lock()

    log_to_stderr()
    logger = get_logger()
    logger.setLevel(logging.INFO)

    add_proc = Process(target=add_500_lock, args=(total, lock))
    print(total.value)
    sub_proc = Process(target=sub_500_lock, args=(total, lock))

    add_proc.start()
    print(total.value)
    sub_proc.start()

    add_proc.join()
    sub_proc.join()
    print(total.value)
