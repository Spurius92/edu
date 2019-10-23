import time
from contextlib import contextmanager


@contextmanager
def timer(msg):
    t0 = time.time()
    print(f'[{msg}] process started.')
    yield
    elapsed_time = time.time() - t0
    print(f'[{msg}] done in {elapsed_time:.5f} sec.')
