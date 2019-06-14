from random import random

class Iterator:
    def __init__(self, k):
        self.k = k
        self.i = 0

    def __next__(self):
        # this is needed to iterate through the object
        if self.i < self.k:
            self.i += 1
            return random()
        else:
            raise StopIteration
    
    def __iter__(self):
        #makes the object iterable
        return self

def random_generator(k):
    for i in range(k):
        yield i

gen = random_generator(5)
for i in gen:
    print(i)
# x = Iterator(3)
# print(next(x))
# print(next(x))
# print(next(x))
# print(next(x))
class DoubeElementListIterator:
    def __init__(self, lst):
        self.lst = lst
        self.i = 0

    def __next__(self):
        # this is needed to iterate through the object
        if self.i < len(self.lst):
            self.i += 2
            return self.lst[self.i - 2], self.lst[self.i - 1]
        else:
            raise StopIteration

class MyList(list):
    def __iter__(self):
        return DoubeElementListIterator(self)


# for pair in MyList([i for i in range(1, 31)]):
#     print(pair)

