# 22 лекция от Хирьянова. Алгоритм Тарьяна. Топологическая сортировка
# по сути то же самое что и обход в глубину(dfs)

# u -число от 1 до N
visited = [False] * (n - 1)
# visited = set()
ans = []


def dfs(start, G, visited, ans):
    visited[start] = True
    for u in G[start]:
        # if not visited[u]:
        if u not in visited:
            dfs(u, G, visited, ans)
    ans.append(start)


# n - number of verteces in graph G
# for i in range(1, n + 1):
#     if not visited[i]:
#         dfs(i, G, visited, ans)

for vertex in G:
    if vertex not in visited:
        dfs(vertex, G, visited, ans)

ans[:] = ans[::-1]