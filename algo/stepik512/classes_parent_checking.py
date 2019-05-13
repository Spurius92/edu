def is_parent(class1, class2, dic):
    if class1 == class2 or class1 in dic[class2]:
        return True
    for i in dic[class2]:
        if i not in dic:
            continue
        if is_parent(class1, i, dic):
            return True
            break
        
n = int(input())
dic = {}
for i in range(n):
    classes = input().split()
    if len(classes) == 1:
        dic[classes[0]] = []
    else:
        dic[classes[0]] = classes[2:]
        
m = int(input())
for i in range(m):
    query = input().split()
    class1 = query[0]
    class2 = query[1]
    if is_parent(class1, class2, dic):
        print('Yes')
    else:
        print('No')
