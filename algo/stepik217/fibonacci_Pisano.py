def fib_mod(n, m):
    a = 0
    b = 1
    Pisano_list = [0, 1]
    Fib_list = [0, 1]
    for i in range(n):
        a, b = b % 10, (a + b) % 10
        Fib_list.append(b)
        Pisano_list.append(b)
    #c = m % 10
    return Pisano_list, Fib_list

def main():
    n, m = map(int, input().split())
    print(fib_mod(n, m))
