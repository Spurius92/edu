import time
from contextlib import contextmanager


@contextmanager
def timer(msg):
    t0 = time.time()
    print(f'[{msg}] process started.')
    yield
    elapsed_time = time.time() - t0
    print(f'[{msg}] done in {elapsed_time:.5f} sec.')


def perform_test(func):
    def wrapper(*args, **kwargs):
        with timer(func.__name__):
            return func(*args, **kwargs)
    return wrapper


# if __name__ == '__main__':
    # perform_test(linear_search(data, target))
