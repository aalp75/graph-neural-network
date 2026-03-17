from queue import PriorityQueue

from graph import Graph

"""
List of implemented algorithms:

- BFS
- Bellman-Ford
- Prim
- DFS
- Dijkstra
"""

def compute_longest_path(graph: Graph) -> float:
    """
    Compute longest shortest path using Floyd-Warshall.
    Complexity is O(graph.num_nodes ^ 3)
    """
    n = graph.num_nodes
    dist = [[float('inf')] * n for _ in range(n)]

    for u in range(n):
        dist[u][u] = 0
        for v, w in graph.adj[u]:
            dist[u][v] = min(dist[u][v], w)

    for k in range(n):
        for u in range(n):
            for v in range(n):
                dist[u][v] = min(dist[u][v], dist[u][k] + dist[k][v])

    longest_path = max(dist[u][v] for u in range(n) for v in range(n) if dist[u][v] < float('inf'))

    return longest_path + 1

def compute_bfs_states(graph: Graph, source: int) -> list:
    state = [0.] * graph.num_nodes
    state[source] = 1.

    states = [state.copy()]

    while True:
        next_state = state.copy()
        for node in range(graph.num_nodes):
            if state[node] == 1.:
                for edge in graph.adj[node]:
                    neigh = edge[0]
                    next_state[neigh] = 1.

                    
        if next_state == state:
            break
        state = next_state
        states.append(state.copy())

    return states

def compute_bf_states(graph: Graph, source:int) -> tuple:

    n = graph.num_nodes

    inf = compute_longest_path(graph) + 1

    state = [inf] * graph.num_nodes # max value is n
    pred = [source] * n

    state[source] = 0.0
    pred[source] = source

    states = [state.copy()]
    preds = [pred.copy()]

    for _ in range(n):
        next_state = state.copy()
        next_pred = pred.copy()
        for node in range(graph.num_nodes):
            for neigh, weight in graph.adj[node]:
                if state[node] + weight < next_state[neigh]:
                    next_state[neigh] = state[node] + weight
                    next_pred[neigh] = node

        if next_state == state:
            break
        state = next_state
        pred = next_pred
        
        states.append(state.copy())
        preds.append(pred.copy())

    return states, preds, inf

def compute_prim_states(graph: Graph, source: int) -> tuple:

    n = graph.num_nodes

    state = [0] * graph.num_nodes # max value is n
    pred = [source] * n

    state[source] = 1
    pred[source] = source

    states = [state.copy()]
    preds = [pred.copy()]

    for _ in range(n):
        next_state = state.copy()
        next_pred = pred.copy()

        best_node = 0 # node not discovered with the smallest edge
        best_pred = 0
        best_weight = 1e10

        for node in range(graph.num_nodes):
            for neigh, weight in graph.adj[node]:
                if state[node] == 1 and state[neigh] == 0 and weight < best_weight:
                    best_weight = weight
                    best_node = neigh
                    best_pred = node

        next_state[best_node] = 1
        next_pred[best_node] = best_pred

        if next_state == state or best_weight == 1e10:
            break
        state = next_state
        pred = next_pred
        
        states.append(state.copy())
        preds.append(pred.copy())

    return states, preds

def compute_dfs_states(graph: Graph, source:int) -> list:

    state = [0.] * graph.num_nodes
    state[source] = 1.
    states = [state.copy()]

    stack = [source]
    visited = {source}

    while stack:
        node = stack[-1]
        found = False
        for neigh, _ in graph.adj[node]:
            if neigh not in visited:
                visited.add(neigh)
                state[neigh] = 1.
                stack.append(neigh)
                states.append(state.copy())
                found = True
                break
        if not found:
            stack.pop()

    return states

def compute_dijkstra_states(graph: Graph, source: int) -> tuple:
    n = graph.num_nodes

    inf = compute_longest_path(graph) + 1
    state = [inf] * n
    pred = [source] * n

    state[source] = 0.0
    pred[source] = source

    states = [state.copy()]
    preds = [pred.copy()]

    visited = [False] * n
    pq = PriorityQueue()
    pq.put((0.0, source))

    while not pq.empty():
        dist, node = pq.get()

        if visited[node]:
            continue
        visited[node] = True

        for neigh, weight in graph.adj[node]:
            if neigh == node:
                continue
            new_dist = state[node] + weight
            if new_dist < state[neigh]:
                state[neigh] = new_dist
                pred[neigh] = node
                pq.put((new_dist, neigh))

        states.append(state.copy())
        preds.append(pred.copy())

    return states, preds, inf

def compute_states(algo: str, graph: Graph, source: int):
    match algo:
        case 'BFS':
            return compute_bfs_states(source, graph), None, None
        case 'DFS':
            return compute_dfs_states(source, graph), None, None
        case 'BF':
            states, preds, inf = compute_bf_states(source, graph)
            return states, preds, inf
        case 'PRIM':
            states, preds = compute_prim_states(source, graph)
            return states, preds, None
        case 'Dijkstra':
            states, preds, inf = compute_dijkstra_states(source, graph)
            return states, preds, inf
        case _:
            raise ValueError(f"Unknown algorithm: {algo}")

def generate_examples(algo: str, graph: Graph, states: list, parents: list | None = None) -> list:
    data = []
    for i in range(len(states) - 1):
        parent = None if parents is None else parents[i + 1]
        data.append((algo, graph, states[i], states[i + 1], parent))
    return data

if __name__ == "__main__":
    ## BFS

    adj = [
        [(0, 1.0), (1, 1.0), (2, 1.0)],
        [(0, 1.0), (1, 1.0), (2, 1.0), (3, 1.0)],
        [(0, 1.0), (1, 1.0), (2, 1.0), (4, 1.0)],
        [(3, 1.0), (1, 1.0)],
        [(4, 1.0), (2, 1.0)]
    ]

    g = Graph(5, adj)
    print(g)

    print("\n-- Breadth First Search --")
    states = compute_bfs_states(0, g)
    ite = 1
    for state in states:
        print(f"{ite} - {state}")
        ite += 1

    print()

    ## Bellman-Ford
    print("\n-- Bellman-Ford --")
    states, predecessors, inf = compute_bf_states(0, g)
    ite = 1
    for state in states:
        print(f"{ite} - {state}")
        ite += 1

    # Prim
    print("\n-- Prim --")
    states, preds = compute_prim_states(0, g)
    ite = 1
    for state in states:
        print(f"{ite} - {state}")
        ite += 1

    # Dijkstra
    print("\n-- Dijkstra --")
    states, preds, inf = compute_dijkstra_states(0, g)
    ite = 1
    for state in states:
        print(f"{ite} - {state}")
        ite += 1