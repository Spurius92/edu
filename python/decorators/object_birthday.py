import time
from fancy_repr_example import better_repr


def birthday(c):
    def wrapper(*args, **kwargs):
        ob = c(*args, **kwargs)
        ob._created_at = time.ctime()
        return ob
    return wrapper


@better_repr
@birthday
class Foo:
    def __init__(self, x, y):
        self.x = x
        self.y = y


f = Foo(10, [20, 30])
# print(f)
print('Created at: ', f._created_at)
