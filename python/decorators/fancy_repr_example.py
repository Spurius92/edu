def fancy_repr(self):
    return f"I am {type(self).__name__}, and i have vars: {vars(self)}"


def better_repr(c):
    c.__repr__ = fancy_repr
    return c


@better_repr
class Foo:
    def __init__(self, x, y):
        self.x = x
        self.y = y


f = Foo(10, [20, 30])
print(f)
