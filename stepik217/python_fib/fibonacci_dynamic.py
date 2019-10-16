def dynamic_fib(n):
    '''
    We only interested in last digits of fibonacci numbers.
    So this is a smart way to calculate them faster and with less memory usage
    '''
    a = 0
    b = 1
    for i in range(n):
        a, b = b % 10, (a + b) % 10
    return b


def main():
    n = int(input())
    print(dynamic_fib(n))

    
if __name__ == '__main__':
    main()
