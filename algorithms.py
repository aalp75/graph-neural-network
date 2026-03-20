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

def simulate_bf(graph: Graph, state:list, predcessor:list) -> tuple:
    next_state = state.copy()
    next_predecessor = predcessor.copy()
    for node in range(graph.num_nodes):
        for neigh, weight in graph.adj[node]:
            if state[node] + weight < next_state[neigh]:
                next_state[neigh] = state[node] + weight
                next_predecessor[neigh] = node

    return next_state, next_predecessor

def compute_bf_states(graph: Graph, source:int) -> tuple:

    max_dist = graph.get_longest_path() + 1

    state = [max_dist] * graph.num_nodes
    predecessor = [-1] * graph.num_nodes
    termination = []

    state[source] = 0.0

    states = [state.copy()]
    predecessors = [predecessor.copy()]

    for _ in range(graph.num_nodes):
        next_state, next_predecessor = simulate_bf(graph, state, predecessor)

        if next_state == state:
            break
        state = next_state
        predecessor = next_predecessor

        states.append(state.copy())
        predecessors.append(predecessor.copy())

    return states, predecessors, max_dist

def compute_prim_states(graph: Graph, source: int) -> tuple:

    n = graph.num_nodes

    state = [0] * graph.num_nodes
    predecessor = [-1] * n

    state[source] = 1

    states = [state.copy()]
    predecessors = [predecessor.copy()]

    for _ in range(n):
        next_state = state.copy()
        next_predecessor = predecessor.copy()

        best_node = -1 # # undiscovered node with the smallest connecting edge
        best_predecessor = 0
        best_weight = float('inf')

        for node in range(graph.num_nodes):
            for neigh, weight in graph.adj[node]:
                if state[node] == 1 and state[neigh] == 0 and weight < best_weight:
                    best_weight = weight
                    best_node = neigh
                    best_predecessor = node

        next_state[best_node] = 1
        next_predecessor[best_node] = best_predecessor

        if next_state == state or best_node == -1:
            break
        state = next_state
        predecessor = next_predecessor

        states.append(state.copy())
        predecessors.append(predecessor.copy())

    return states, predecessors

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
    """
    Returns states, predecessors, longest path and termination
    """
    match algo:
        case 'bfs':
            states = compute_bfs_states(graph, source)
            termination = [0.0 for i in range(len(states))]
            termination[-1] = 1.0
            return states, None, None, termination
        case 'bf':
            states, predecessors, max_dist = compute_bf_states(graph, source)
            termination = [0.0 for i in range(len(states))]
            termination[-1] = 1.0
            return states, predecessors, max_dist, termination
        case 'prim':
            states, predecessors = compute_prim_states(graph, source)
            termination = [0.0 for i in range(len(states))]
            termination[-1] = 1.0
            max_dist = 30
            return states, predecessors, max_dist, termination
        case 'cc':
            states = compute_cc_states(graph)
            termination = [0.0 for i in range(len(states))]
            termination[-1] = 1.0
            return states, None, None, termination
        case _:
            raise ValueError(f"Unknown algorithm: {algo}")

def generate_examples(states: list, predecessors: list | None = None, terminations: list | None = None) -> list:
    data = []
    for i in range(len(states) - 1):
        predecessor = None if predecessors is None else predecessors[i + 1]
        termination = None if terminations is None else terminations[i + 1]
        data.append((states[i], states[i + 1], predecessor, termination))
    return data

if __name__ == "__main__":

    g = Graph(5)

    g.add_edge(0, 1, 1.0)
    g.add_edge(0, 2, 1.0)
    g.add_edge(1, 2, 1.0)
    g.add_edge(1, 3, 1.0)
    g.add_edge(2, 4, 1.0)

    print(g)

    ## BFS
    print("\nBreadth First Search algorithm:")
    states = compute_bfs_states(g, 0)
    for ite, state in enumerate(states):
        print(f"Step {ite}: {state}")

    print()

    ## Bellman-Ford
    print("\nBellman-Ford algorithm:")
    states, predecessors, max_dist = compute_bf_states(g, 0)
    for ite, state in enumerate(states):
        print(f"Step {ite}: {state}")

    # Prim
    print("\nPrim algorithm:")
    states, predecessors = compute_prim_states(g, 0)
    for ite, state in enumerate(states):
        print(f"Step {ite}: {state}")