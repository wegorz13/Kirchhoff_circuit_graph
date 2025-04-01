import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import base_cycles as bc

def create_graph(mode, size, max_resistance, max_power):
    match mode:
        case "e":
            G = nx.erdos_renyi_graph(size,0.5)
        case "c":
            G = nx.cubical_graph(size,0.5)

    circuit = (np.random.randint(0,size//2), np.random.randint(size//2+1, size), np.random.randint(1, max_power+1) )

    nx.draw(G, with_labels=True, font_weight='bold')
    plt.show()

    return create_graph_matrix(size, add_weight_to_edges(G.edges(), max_resistance)), circuit, list(G.edges()), G

def add_weight_to_edges(edges, max_resistance):
    w_edges = []
    for u,v in edges:
        weight = np.random.randint(1, max_resistance+1)
        w_edges.append((u, v, weight))
    return w_edges

def create_graph_matrix(n, edges):
    G = [[0 for _ in range(n)] for _ in range(n)]
    for u,v,w in edges:
        G[u][v] = w
        G[v][u] = w
    return G

def map_edges(n,edges):
    i = 0
    edge_map = [[-1 for _ in range(n)] for _ in range(n)]
    for s,t in edges:
        edge_map[s][t] = i
        edge_map[t][s] = i
        i += 1
    return edge_map

def solver(G, C, E, m, n):
    edge_map = map_edges(n, E)
    S = [[0 for _ in range(m)] for _ in range(m)]
    B = [0 for _ in range(m)]
    start = C[0]
    end = C[1]
    UV = C[2]

    for i in range(n):
        if G[start][i] != 0:
            S[0][edge_map[start][i]] = 1
        if G[end][i] != 0:
            S[0][edge_map[end][i]] = -1

    ind=1
    for v in range(n):
        if v!=start and v!=end:
            for t in range(n):
                if G[v][t] != 0:
                    if t==start:
                        S[ind][edge_map[v][t]] = -1
                    elif t==end:
                        S[ind][edge_map[v][t]] = 1
                    elif v<t:
                        S[ind][edge_map[v][t]] = 1
                    else:
                        S[ind][edge_map[v][t]] = -1
            ind+=1

    base_cycles = bc.base_cycles(E)

    flag=False
    for i in range(m-n+1):
        cycle = base_cycles[i]

        if not flag and start in cycle and end in cycle:
            flag=True
            B[ind]=UV
        if not flag and i==m-n:
            break

        for k in range(len(cycle)-1):
            if cycle[k]<cycle[k+1]:
                S[ind][edge_map[cycle[k]][cycle[k+1]]] = G[cycle[k]][cycle[k+1]]
            else:
                S[ind][edge_map[cycle[k]][cycle[k + 1]]] = -G[cycle[k]][cycle[k + 1]]
        if cycle[0] < cycle[len(cycle)-1]:
            S[ind][edge_map[cycle[0]][cycle[len(cycle)-1]]] = G[cycle[0]][cycle[len(cycle)-1]]
        else:
            S[ind][edge_map[cycle[0]][cycle[len(cycle) - 1]]] = -G[cycle[0]][cycle[len(cycle) - 1]]
        ind+=1

    for r in S:
        print(r)

    adj = [[] for _ in range(n)]

    for u, v in E:
        adj[u].append(v)
        adj[v].append(u)
    if not flag:
        source_cycle = bc.bfs(start,end,adj)
        if source_cycle:
            B[ind]=UV
            for k in range(len(source_cycle) - 1):
                if source_cycle[k]<source_cycle[k+1]:
                    S[ind][edge_map[source_cycle[k]][source_cycle[k + 1]]] = G[source_cycle[k]][source_cycle[k + 1]]
                else:
                    S[ind][edge_map[source_cycle[k]][source_cycle[k + 1]]] = -G[source_cycle[k]][source_cycle[k + 1]]

    A = np.array(S)
    B = np.array(B)

    solved = np.linalg.solve(A,B)
    print(solved)
    solution = [[0 for _ in range(n)] for _ in range(n)]

    for i in range(n):
        for j in range(i,n):
            if edge_map[i][j] != -1:
                solution[i][j] = solved[edge_map[i][j]]
                solution[j][i] = -solved[edge_map[i][j]]

    return solution


def draw_solution_graph(G_orig, solution, circuit):
    n = len(solution)
    start, end = circuit[0], circuit[1]

    G_sol = nx.DiGraph()
    for i in range(n):
        G_sol.add_node(i)

    edge_weights = []
    for i in range(n):
        for j in range(i,n):
            if solution[i][j] != 0:
                if solution[i][j] > 0:
                    G_sol.add_edge(i, j, weight=solution[i][j])
                    edge_weights.append(solution[i][j])
                else:
                    G_sol.add_edge(j, i, weight=-solution[i][j])
                    edge_weights.append(-solution[i][j])

    cmap = plt.cm.viridis
    if edge_weights:
        norm = plt.Normalize(min(edge_weights), max(edge_weights))
        edge_colors = [cmap(norm(data['weight'])) for u, v, data in G_sol.edges(data=True)]
    else:
        edge_colors = "black"

    pos = nx.spring_layout(G_sol)

    plt.figure(figsize=(8, 6))
    nx.draw_networkx_edges(G_sol, pos, edge_color=edge_colors, arrows=True, arrowstyle='->', arrowsize=50)
    for node in G_sol.nodes():
        node_color = "yellow" if node in [start, end] else "lightblue"
        lw = 3.0 if node in [start, end] else 1.0
        nx.draw_networkx_nodes(G_sol, pos,
                               nodelist=[node],
                               node_color=[node_color],
                               node_size=800,
                               edgecolors='black',
                               linewidths=lw)

    nx.draw_networkx_labels(G_sol, pos)

    edge_labels = {(u, v): round(data['weight'], 2) for u, v, data in G_sol.edges(data=True)}
    nx.draw_networkx_edge_labels(G_sol, pos, edge_labels=edge_labels)

    plt.title("Solution Graph with Directed Edges and Colored by Weight")
    plt.axis("off")
    plt.show()

G_matrix, C, E, G_orig = create_graph("e", 10, 5, 15)
solution = solver(G_matrix, C, E, len(E), 10)

print("Solution Matrix:")
for row in solution:
    print(row)

draw_solution_graph(G_orig, solution, C)

