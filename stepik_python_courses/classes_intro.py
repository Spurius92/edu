class MoneyBox:
    '''create moneybox, check its capacity and add coins if possible
    '''
    def __init__(self, capacity):
        self.count = 0
        self.capacity = capacity
        
    def can_add(self, v):
        return self.capacity >= self.count + v
    
    def add(self, v):
        self.count += v


        
class Buffer:
    '''
    add arbitrary number od values to list. If lenght of a list reaches 5,
    print sum of them and clear the list for new values
    '''
    def __init__(self):
        self.lst = []

    def add(self, *a):
        for i in a:
            self.lst.append(i)
            if len(self.lst) == 5:
                print(sum(self.lst))
                self.lst.clear()

    def get_current_part(self):
        return self.lst
