# from test_speed import perform_test
from test_speed import timer

data = list(range(1000000))
target = 58
low = 0
high = len(data) - 1


def linear_search():
    for i in range(len(data)):
        if data[i] == target:
            print(i)
            return True
    return False


def binary_search_iterative():
    low = 0
    high = len(data) - 1
    while low <= high:
        middle = (low + high) // 2
        if target == data[middle]:
            print(middle)
            return True
        elif target > data[middle]:
            low = middle + 1
        else:
            high = middle - 1
    return False


def binary_search_recursive():
    global data, target, low, high
    if low > high:
        return False
    else:
        middle = (low + high) // 2
        if target == data[middle]:
            print(middle)
            return True
        elif target > data[middle]:
            return binary_search_recursive()
        else:
            return binary_search_recursive()


if __name__ == '__main__':
    timer(binary_search_iterative)
        # print(linear_search(data, target))
        # print(binary_search_iterative(data, target))
        # print(binary_search_recursive(data, target, 0, len(data) - 1))
