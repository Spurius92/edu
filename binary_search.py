import random

a = 1
b = 1001
secret = random.randrange(a, b)
count = 0
print('Starting with: ', a, b, secret)

while a < b:
    m = (a + b) // 2
    print(m)
    if secret >= m:
        a = m
    else:
        b = m
    count += 1
    if count == 11:
        print('reached 11')
        break
print(secret, a)
