from graph import Graph
import random

def graph_generation(
        num_nodes: int, 
        p: float = 0.5,
        seed: int | None = None, 
        weighted: bool = False,
        connected: bool = True, # unused for now
        self_loop: bool = True,
        weight_mn: float = 0,
        weight_mx: float = 1,
) -> Graph:

    if seed is not None:
        random.seed(seed)

    adj = [[] for _ in range(num_nodes)]

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if random.random() < p:
                w = 1.0
                if weighted:
                    w = random.uniform(weight_mn, weight_mx)
                adj[i].append((j, w))
                adj[j].append((i, w))

    if self_loop:
        for i in range(num_nodes):
            adj[i].append((i, 0.0))

    return Graph(num_nodes, adj)


if __name__ == "__main__":
    num_nodes = 5
    p = 0.5
    g = graph_generation(num_nodes,
                         p, 
                         weighted=True,
                         connected=True,
                         self_loop=True,
                         weight_mn=0,
                         weight_mx=2
    )
    print(g)

