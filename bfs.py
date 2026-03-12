import graph

"""
Algorithms to learn
"""

def bfs_states(source, graph):
    state = [0.] * graph.num_nodes
    state[source] = 1.

    states = [state.copy()]

    while True:
        next_state = state.copy()
        for node in range(graph.num_nodes):
            if state[node] == 1.:
                for neigh in graph.adj[node]:
                    next_state[neigh] = 1.

                    
        if next_state == state:
            break
        state = next_state
        states.append(state.copy())

    return states

def generate_training_examples(g, states):
    data = []
    for i in range(len(states) - 1):
        data.append((g, states[i], states[i + 1]))
    return data

if __name__ == "__main__":
    adj = [
        [0, 1, 2],
        [0, 1, 2, 3],
        [0, 1, 2, 4],
        [3, 1],
        [4, 2]
    ]

    g = graph.Graph(5, adj)
    print(g)

    states = bfs_states(0, g)
    ite = 1
    for state in states:
        print(f"{ite} - {state}")
        ite += 1