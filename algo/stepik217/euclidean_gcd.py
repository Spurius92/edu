def euclidean_algorithm(a, b):
    '''euclidean algorithm for finding greatest common divisor of two numbers.
    recursively find gcd for the smaller number and remainder of division greater by smaller
    '''
    if a == 0:
        return b
    elif b == 0:
        return a
    elif a >= b:
        return euclidean_algorithm(a % b, b)
    else:
        return euclidean_algorithm(a, b % a)

def main():
    a, b = map(int, input().split())
    print(euclidean_algorithm(a, b))

if __name__ == '__main__':
    main()
               
