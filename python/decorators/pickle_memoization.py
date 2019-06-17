import time
import pickle

from timer import timer

"""
https://www.youtube.com/watch?v=MjHpMCIvwsY
"""


def memoize(func):
    cache = {}

    def wrapper(*args, **kwargs):
        t = (pickle.dumps(args), pickle.dumps(kwargs))
        if t not in cache:
            print(f"Caching NEW values for {func.__name__}{args}")
            cache[t] = func(*args, **kwargs)
        else:
            print(f"Using OLD values for {func.__name__}{args}")
        return cache[t]
    return wrapper


@memoize
def add(a, b):
    print('Running add')
    time.sleep(1)
    return a + b


@memoize
def mul(a, b):
    print('Running mul')
    time.sleep(1)
    return a * b


if __name__ == '__main__':
    with timer('running functions with pickled memoization'):
        print(add(3, 4))
        print(mul(3, 4))
        print(add(3, 4))
        print(add(3, 4))
