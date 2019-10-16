def fib(n):
    table = [0, 1]
    for i in range(2, n + 1):
        table.append(table[i - 1] + table[i - 2])
    return table[-1]


def main():
    n = int(input())
    print(fib(n))


if __name__ == "__main__":
    main()
