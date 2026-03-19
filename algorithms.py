from graph import Graph

"""
Algorithms:

- Breadth-First-Search (BFS)
- Bellman-Ford (BF)
- Prim
- Connected Components (CC)
"""

def simulate_bfs(graph: Graph, state: list) -> list:
    next_state = state.copy()
    for node in range(graph.num_nodes):
        if state[node] == 1.:
            for neigh, weight in graph.adj[node]:
                next_state[neigh] = 1

    return next_state

def compute_bfs_states(graph: Graph, source: int) -> list:
    state = [0 if i != source else 1 for i in range(graph.num_nodes)]

    states = [state.copy()]

    while True:
        next_state = simulate_bfs(graph, state)

        if next_state == state:
            break
        state = next_state
        states.append(state.copy())

    return states

def simulate_bf(graph: Graph, state:list, parent:list) -> tuple:
    next_state = state.copy()
    next_parent = parent.copy()
    for node in range(graph.num_nodes):
        for neigh, weight in graph.adj[node]:
            if state[node] + weight < next_state[neigh]:
                next_state[neigh] = state[node] + weight
                next_parent[neigh] = node

    return next_state, next_parent

def compute_bf_states(graph: Graph, source:int) -> tuple:

    inf = graph.get_longest_path() + 1

    state = [inf] * graph.num_nodes
    parent = [source] * graph.num_nodes

    state[source] = 0.0
    parent[source] = source

    states = [state.copy()]
    parents = [parent.copy()]

    for _ in range(graph.num_nodes):
        next_state, next_parent = simulate_bf(graph, state, parent)

        if next_state == state:
            break
        state = next_state
        parent = next_parent

        states.append(state.copy())
        parents.append(parent.copy())

    return states, parents, inf

def compute_prim_states(graph: Graph, source: int) -> tuple:

    n = graph.num_nodes

    state = [0] * graph.num_nodes
    parent = [source] * n

    state[source] = 1
    parent[source] = source

    states = [state.copy()]
    parents = [parent.copy()]

    for _ in range(n):
        next_state = state.copy()
        next_parent = parent.copy()

        best_node = -1 # # undiscovered node with the smallest connecting edge
        best_parent = 0
        best_weight = float('inf')

        for node in range(graph.num_nodes):
            for neigh, weight in graph.adj[node]:
                if state[node] == 1 and state[neigh] == 0 and weight < best_weight:
                    best_weight = weight
                    best_node = neigh
                    best_parent = node

        next_state[best_node] = 1
        next_parent[best_node] = best_parent

        if next_state == state or best_node == -1:
            break
        state = next_state
        parent = next_parent
        
        states.append(state.copy())
        parents.append(parent.copy())

    return states, parents

def compute_cc_states(graph: Graph) -> list:
    n = graph.num_nodes
    state = [float(i) for i in range(n)]
    states = [state.copy()]

    while True:
        next_state = state.copy()
        for node in range(n):
            for neigh, _ in graph.adj[node]:
                next_state[node] = min(next_state[node], state[neigh])
        if next_state == state:
            break
        state = next_state
        states.append(state.copy())

    return states

def compute_states(algo: str, graph: Graph, source: int) -> tuple:
    match algo:
        case 'BFS':
            return compute_bfs_states(graph, source), None, None, None
        case 'BF':
            states, parents, inf = compute_bf_states(graph, source)
            return states, parents, inf, None
        case 'PRIM':
            states, parents = compute_prim_states(graph, source)
            return states, parents, None, None
        case 'CC':
            return compute_cc_states(graph), None, None, None
        case _:
            raise ValueError(f"Unknown algorithm: {algo}")

def generate_examples(states: list, parents: list | None = None) -> list:
    data = []
    for i in range(len(states) - 1):
        parent = None if parents is None else parents[i + 1]
        data.append((states[i], states[i + 1], parent))
    return data

if __name__ == "__main__":
    
    adj = [
        [(0, 1.0), (1, 1.0), (2, 1.0)],
        [(0, 1.0), (1, 1.0), (2, 1.0), (3, 1.0)],
        [(0, 1.0), (1, 1.0), (2, 1.0), (4, 1.0)],
        [(3, 1.0), (1, 1.0)],
        [(4, 1.0), (2, 1.0)]
    ]

    g = Graph(5, adj)
    print(g)

    ## BFS
    print("\n-- Breadth First Search --")
    states = compute_bfs_states(g, 0)
    for ite, state in enumerate(states):
        print(f"{ite} - {state}")

    print()

    ## Bellman-Ford
    print("\n-- Bellman-Ford --")
    states, parents, inf = compute_bf_states(g, 0)
    for ite, state in enumerate(states):
        print(f"{ite} - {state}")

    # Prim
    print("\n-- Prim --")
    states, parents = compute_prim_states(g, 0)
    for ite, state in enumerate(states):
        print(f"{ite} - {state}")