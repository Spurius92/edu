from itertools import takewhile
import time
        
"""
Решение по теореме Вильсона. 
Для числа p, если (p-1)! + 1 делится на само число без остатка, то число простое.
Генератор построен аналогично функции для нахождения чисел Фибоначчи.
Есть само ичисло и факториал предыдущего.
Далее мы проверяем делится ли (факториал + 1) на число
и если делится, выдаем число в генераторе.
После этого обновляем факториал и число

При тестировании получилось, что небольшое число 182777 проверялось уже 185 секунд...
Нельзя ставить на большие числа
Spent time for 182777 number:  185.0377550125122 seconds
"""
def primes():
    num, fact = 2, 1
    while True:
        if (fact + 1) % num == 0:
            yield num
        fact, num = fact * num, num + 1


def test_primes(check):
    start = time.time()
    print(list(takewhile(lambda x : x <= check, primes())))
    # list(takewhile(lambda x : x <= check, primes()))
    end = time.time()
    print(f'Spent time for {check} number: ', str(end-start), 'seconds')

def main():
    test_primes(182777)

if __name__ == '__main__':
    main()