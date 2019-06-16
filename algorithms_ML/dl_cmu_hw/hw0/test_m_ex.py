import numpy as np
import os
import sys
import time
import torch    

import sys
sys.path.insert(0, './hw0')

import hw0

x = np.array([np.arange(4 * i).reshape(i, -1) for i in range(3,8)])
print(x.shape)
print(x)

l = 2
sp = 1

print('\nhw0.slice_fixed_point(x, {}, {})'.format(l, sp))
print(hw0.slice_fixed_point(x, l, sp))

print('\nhw0.slice_last_point(x, {})'.format(l))
print(hw0.slice_last_point(x, l))

seed = 8
np.random.seed(seed)
print('\nnp.random.seed({})'.format(seed))
print('\nhw0.slice_random_point(x, {})'.format(l))
print(hw0.slice_random_point(x, l))


print('\nhw0.pad_pattern_end(x)')
print(hw0.pad_pattern_end(x))

print('\nhw0.pad_constant_central(x, -1)')
print(hw0.pad_constant_central(x, -1))
