space = {'global': {'parent': None, 'vars': []}}

def create(namespace, parent):
    space[namespace] = {'parent': parent, 'vars': []}
        
def add(namespace, var):
    space[namespace]['vars'].append(var) 
    
def get(namespace, var):
    if var in space[namespace]['vars']:
        print(namespace)

    else:
        if space[namespace]['parent'] is None:
            print(None)
            return
        get(space[namespace]['parent'], var)
    
n = int(input())
k_query = 0
while k_query != n:
    query = input().split()
    
    if query[0] == 'create':
        create(query[1], query[2])
    elif query[0] == 'add':
        add(query[1], query[2])
    elif query[0] == 'get':
        get(query[1], query[2])
        
    k_query += 1
