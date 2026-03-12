import graph
import random

def graph_generation(
        n: int, 
        p: float = 0.5,
        seed: int | None = None, 
        connected: bool = True, 
        self_loop: bool =True
) -> graph.Graph:

    if seed is not None:
        random.seed(seed)

    adj = [[] for _ in range(n)]

    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < p:
                adj[i].append(j)
                adj[j].append(i)

    if self_loop:
        for i in range(n):
            adj[i].append(i)

    return graph.Graph(n, adj)


if __name__ == "__main__":
    g = graph_generation(5, 0.5, None, connected=True, self_loop=True)
    print(g)

