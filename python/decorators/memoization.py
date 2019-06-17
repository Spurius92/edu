import time


def memoize(func):
    cache = {}

    def wrapper(*args, **kwargs):
        if args not in cache:
            print(f"Caching NEW values for {func.__name__}{args}")
            cache[args] = func(*args, **kwargs)
        else:
            print(f"Using OLD values for {func.__name__}{args}")
        return cache[args]
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
    print(add(3, 4))
    print(mul(3, 4))
    print(add(3, 4))
