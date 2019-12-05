import numpy as np
import random
from scipy import optimize

SEED = 17
random.seed(SEED)
np.random.seed(SEED)
x = np.array([[-2, -1, 0, 2], [0, -2, -1, 1]])
# ys = np.array()


def z(x):
    return x[0] / ((x[0] ** 2) + 2 * (x[1] ** 2) + 1)


def generate_new(x, sigma):
    maximums = np.argmax(z(x))
    # print(z(x))
    print(maximums)
    new_x = np.random.normal(x[0][maximums], sigma, x[0].shape)
    new_y = np.random.normal(x[1][maximums], sigma, x[1].shape)
    # print('new:', new_x, new_y)
    return np.vstack((new_x, new_y))


# for i in range(1, 5):
#     x = generate_new(x, .25 * 1 / i)
#     print(x, '\n')

# print('final: ', '\n', x)
# maximums = (np.argsort(z(x)))
# print(random.choices(x[1], weights=maximums))
# print(optimize.minimize(z, x0=[[-2], [0]]))
print(np.array([-9.99999918e-01, -7.09815077e-09]))