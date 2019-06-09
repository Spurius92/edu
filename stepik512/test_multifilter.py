from multifilter import multifilter

def mul2(x):
    return x % 2 == 0

def mul3(x):
    return x % 3 == 0

def mul5(x):
    return x % 5 == 0


a = [i for i in range(31)] # [0, 1, 2, ... , 30]

# print(list(multifilter(a, mul2, mul3, mul5))) 
print(list(multifilter(a, mul2, mul3, mul5, judge=multifilter.judge_half))) 
# print(list(multifilter(a, mul2, mul3, mul5, judge=multifilter.judge_all))) 
