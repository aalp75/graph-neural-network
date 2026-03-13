from graph import Graph

"""
Algorithms to learn:

- BFS
- Bellman-Ford
- DFS: ongoing
"""

def compute_bfs_states(source: int, graph: Graph) -> list:
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

def compute_bellman_ford_states(source:int , graph: Graph) -> list:

    n = graph.num_nodes

    state = [n] * graph.num_nodes # max value is n
    state[source] = 0.

    states = [state.copy()]

    for _ in range(n):
        next_state = state.copy()
        for node in range(graph.num_nodes):
            for edge in graph.adj[node]:
                neigh, weight = edge[0], edge[1]
                next_state[neigh] = min(next_state[neigh], state[node] + weight)
                    
        if next_state == state:
            break
        state = next_state
        states.append(state.copy())

    return states


def compute_dfs_states(source:int , graph: Graph) -> list:

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

def generate_examples(algo: str, graph: Graph, states: list) -> list:
    data = []
    for i in range(len(states) - 1):
        data.append((algo, graph, states[i], states[i + 1]))
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

    states = compute_bfs_states(0, g)
    ite = 1
    for state in states:
        print(f"{ite} - {state}")
        ite += 1

    print()

    ## Bellman-Ford
    states = compute_bellman_ford_states(0, g)
    ite = 1
    for state in states:
        print(f"{ite} - {state}")
        ite += 1