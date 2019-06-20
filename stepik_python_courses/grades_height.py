import math

# input_data = [
#     [6, 'Вяххи', 159],
#     [11, 'Федотов', 172],
#     [7, 'Бондарев', 158],
#     [6, 'Чайкина', 153]]
with open('E:/dataset_3380_5.txt', 'r') as dataset:
    data = [d.strip().split('\t') for d in dataset]


table = {i: [] for i in range(1, 12)}

for i in data:
    grade, _, height = i
    grade, height = int(grade), int(height)
    # print(grade, _, height)
    if not grade:
        break
    if table[grade]:
        table[grade].append(height)
    else:
        table[(grade)] = [height]

print(table)
for k, v in table.items():
    if v != '-':
        table[k] = round(math.fsum(v) / len(v), 5)
        print(k, table[k])
