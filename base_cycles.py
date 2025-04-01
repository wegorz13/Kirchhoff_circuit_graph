from collections import deque

class Node:
    def __init__(self,value):
        self.val=value
        self.parent=self
        self.rank=0

def find(x):
    if x.parent!=x:
        x.parent = find(x.parent)
    return x.parent

def union(X,Y):
    X = find(X)
    Y = find(Y)

    if X==Y:
        return

    if X.rank > Y.rank:
        Y.parent = X
    elif X.rank < Y.rank:
        X.parent = Y

    else:
        X.parent=Y
        Y.rank+=1

def spanning_tree(E):
    nodes = {}
    converted_edges = []
    for u, v in E:
        if u not in nodes:
            nodes[u] = Node(u)
        if v not in nodes:
            nodes[v] = Node(v)
        converted_edges.append((nodes[u], nodes[v]))

    Tree = []
    Free_edges = []
    for u, v in converted_edges:
        if find(u) != find(v):
            Tree.append((u.val, v.val))
            union(u, v)
        else:
            Free_edges.append((u.val, v.val))
    return Tree, Free_edges

def bfs(start, end, adj):
    n=len(adj)
    visited = [False for _ in range(n)]
    queue = deque()
    queue.append((start, [start]))

    while queue:
        node, path = queue.popleft()
        if node == end:
            return path
        for neighbor in adj[node]:
            if not visited[neighbor]:
                visited[neighbor]=True
                queue.append((neighbor, path + [neighbor]))
    return None

def base_cycles(Edges):
    Tree, Free_edges = spanning_tree(Edges)

    max_node = max(max(u, v) for u, v in Edges) if Edges else 0
    adj = [[] for _ in range(max_node+1)]
    for u, v in Tree:
        adj[u].append(v)
        adj[v].append(u)

    cycles = []
    for u, v in Free_edges:
        path = bfs(u, v, adj)
        if path:
            cycle = path + [u]
            cycles.append(cycle)

    return cycles


