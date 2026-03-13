from graph import Graph
import random

def graph_generation(
        n: int, 
        p: float = 0.5,
        seed: int | None = None, 
        weighted: bool = False,
        connected: bool = True, # unused for now
        self_loop: bool = True,
        weight_range: float = 1.0
) -> Graph:

    if seed is not None:
        random.seed(seed)

    adj = [[] for _ in range(n)]

    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < p:
                w = 1.
                if weighted:
                    w = random.uniform(0, weight_range)
                adj[i].append((j, w))
                adj[j].append((i, w))

    if self_loop:
        for i in range(n):
            adj[i].append((i, 0))

    return Graph(n, adj)


if __name__ == "__main__":
    g = graph_generation(5, 0.5, None, weighted=False, connected=True, self_loop=True)
    print(g)

