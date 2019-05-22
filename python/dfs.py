# Обход графа в глубину 
# лекция 22 от Хирьянова
# "Call all friends"

# G = graph. Dictiioary with pairs of verteces and their coordinates
def dfs(vertex, G, used):
	# used = used or set()
	used.add(vertex)
	for neighbor in G[vertex]:
		if neighbor not in used:
			dfs(neighbor, G, used) 


used = {}
N = 0

for vertex in G:
    if vertex not in G:
        dfs(vertex, G, used)
        N += 1
print(N)