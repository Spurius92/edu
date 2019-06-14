"""
This is a file for stepik course 'introduction to python'
not finished

"""

from collections import Counter
# freq = {}
with open('E:\dataset_3363_3.txt', 'r') as dataset:
    # lower_words = dataset.read().upper().strip().split(' ')
    # freq = Counter(lower_words)

    # # print(freq)
    # print(freq.most_common()[:10])
    # print('*' * 30)

    words = dataset.read().strip().split(' ')
    freq1 = Counter(words)
    most = freq1.most_common(1)[0][1]
    print(most)
    for el in freq1.elements():
        pass
    # print(freq1)
    # print(freq1.most_common()[:10])
