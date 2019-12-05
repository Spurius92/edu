# source: https://docs.python.org/3/library/functools.html
from functools import lru_cache

example = """
    @lru_cache(maxsize=None)
    def fib(n):
        if n < 2:
            return n
        return fib(n-1) + fib(n-2)

    >>> [fib(n) for n in range(16)]
    [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610]

    >>> print(fib.cache_info())
    CacheInfo(hits=28, misses=16, maxsize=None, currsize=16)

    >>> fib.cache_clear()
    >>> print(fib.cache_info())
    CacheInfo(hits=0, misses=0, maxsize=128, currsize=0)
"""


@lru_cache(maxsize=128)
def fib(n):
    if n < 2:
        return n
    return fib(n - 1) + fib(n - 2)


fibs = [fib(n) for n in range(160)]
print(fib.cache_info())
fib.cache_clear()
print(fib.cache_info())