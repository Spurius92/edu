from random import choices
from math import log2

'''
Interesting example with implementation of cross entropy
Reference: https://dementiy.github.io/2018/07/05/crossentropy/
'''

probabilities_1 = {
    'A': 0.25,
    'B': 0.25,
    'C': 0.25,
    'D': 0.25,
}

probabilities_2 = {
    'A': 0.5,
    'B': 0.125,
    'C': 0.125,
    'D': 0.25,
}

sequence1 = choices(
    population=list(probabilities_1.keys()),
    weights=probabilities_1.values(),
    k=100
)

sequence2 = choices(
    population=list(probabilities_2.keys()),
    weights=probabilities_2.values(),
    k=100
)

if __name__ == '__main__':
    print(sum(map(lambda ch: log2(1 / probabilities_1[ch]), sequence1)))
    print(sum(map(lambda ch: log2(1 / probabilities_2[ch]), sequence2)))
