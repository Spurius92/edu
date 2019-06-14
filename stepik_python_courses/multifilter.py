
class multifilter:
    def __init__(self, iterable, *funcs, judge='multifilter.judge_any'):
        self.iterable = iterable
        self.funcs = funcs
        self.length = len(funcs)
        self.judge = judge
        self.table = [[int(func(i)) for i in self.iterable] for func in self.funcs]
        self.pos = [0] * len(self.iterable)
        if len(self.table) > 1:
            for table in range(len(self.table)):
                for pos in range(len(self.iterable)):
                    self.pos[pos] += self.table[table][pos]          
        # print(self.pos)

    def judge_half(self, pos):
        half = self.pos[self.length//2:]
        for p in range(len(self.iterable)):
            if self.pos[p] < (self.length // 2):
                self.iterable[p] *= 0
        
        self.iterable = list(filter(lambda x: x > 0, self.iterable))
        self.iterable = [0] + self.iterable
        print('judge half worked: ' + str(self.iterable))
        yield self.iterable

    def judge_any(self, pos):
        for p in range(len(self.iterable)):
            if self.pos[p] == 0:
                self.iterable[p] *= 0
        
        self.iterable = list(filter(lambda x: x > 0, self.iterable))
        self.iterable = [0] + self.iterable
        # print('judge any worked: ' + str(self.iterable))
        yield self.iterable

    def judge_all(self, pos):
        for p in range(len(self.iterable)):
            if self.pos[p] != self.length:
                self.iterable[p] *= 0
        
        self.iterable = list(filter(lambda x: x > 0, self.iterable))
        self.iterable = [0] + self.iterable
        # print('judge all worked: ' + str(self.iterable))
        yield self.iterable

    def __iter__(self):
        if self.judge == multifilter.judge_all:
            return self.judge_all(self)
        elif self.judge == multifilter.judge_half:
            return self.judge_half(self)
        else:
            return self.judge_any(self)
