from test_speed import perform_test

data = list(range(100))
target = 58


@perform_test
def linear_search(data, target):
    for i in range(len(data)):
        if data[i] == target:
            return True
    return False


@perform_test
def binary_search_iterative(data, target):
    low = 0
    high = len(data) - 1
    while low <= high:
        middle = (low + high) // 2
        if target == data[middle]:
            return True
        elif target > data[middle]:
            low = middle + 1
        else:
            high = middle - 1
    return False


def binary_search_recursive(data, target, low, high):
    if low > high:
        return False
    else:
        middle = (low + high) // 2
        if target == data[middle]:
            return True
        elif target > data[middle]:
            return binary_search_recursive(data, target, middle + 1, high)
        else:
            return binary_search_recursive(data, target, low, middle - 1)


if __name__ == '__main__':
    # print(linear_search(data, target))
    # print(binary_search_iterative(data, target))
    print(binary_search_recursive(data, target, 0, len(data) - 1))
