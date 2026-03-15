import math

from graph import Graph
import random

def add_self_loop(graph: Graph, weighted, weight_mn, weight_mx):

    for i in range(graph.num_nodes):
        w = 1.0
        if weighted:
            w = random.uniform(weight_mn, weight_mx)
        graph.add_edge(i, i, w)

    return graph


def ladder_graph(
    num_nodes: int, 
    seed: int | None = None, 
    weighted: bool = False,
    weight_mn: float = 0,
    weight_mx: float = 1,
    self_loop: bool = True,
) -> Graph:
    
    if seed is not None:
        random.seed(seed)
    
    graph = Graph(num_nodes)
    for node in range(num_nodes):
        if node % 2 == 0:
            if node + 1 < num_nodes:
                w = 1.0 if not weighted else random.uniform(weight_mn, weight_mx)
                graph.add_edge(node, node + 1, w)
            if node + 2 < num_nodes:
                w = 1.0 if not weighted else random.uniform(weight_mn, weight_mx)
                graph.add_edge(node, node + 2, w)
        if node % 2 == 1 and node + 2 < num_nodes:
            w = 1.0 if not weighted else random.uniform(weight_mn, weight_mx)
            graph.add_edge(node, node + 2, w)

    if self_loop:
        add_self_loop(graph, weighted, weight_mn, weight_mx)

    return graph

def grid_graph(
    num_nodes: int, 
    seed: int | None = None, 
    weighted: bool = False,
    weight_mn: float = 0,
    weight_mx: float = 1,
    self_loop: bool = True,
) -> Graph:
    
    if seed is not None:
        random.seed(seed)
        
    graph = Graph(num_nodes)

    c = int(math.sqrt(num_nodes))

    for node in range(num_nodes):
        if node + c < num_nodes:
            w = 1.0 if not weighted else random.uniform(weight_mn, weight_mx)
            graph.add_edge(node, node + c, w)
        if node % c != c - 1 and node + 1 < num_nodes:
            w = 1.0 if not weighted else random.uniform(weight_mn, weight_mx)
            graph.add_edge(node, node + 1, w)

    if self_loop: 
        add_self_loop(graph, weighted, weight_mn, weight_mx)
    
    return graph

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
    
    """
    Erdos-Renyi model to generate random graph
    """

    if seed is not None:
        random.seed(seed)

    graph = Graph(num_nodes)

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if random.random() < p:
                w = 1.0 if not weighted else random.uniform(weight_mn, weight_mx)
                graph.add_edge(i, j, w)

    if self_loop: 
        add_self_loop(graph, weighted, weight_mn, weight_mx)

    return graph

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

