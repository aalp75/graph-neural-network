import math

from graph import Graph
import random

def random_weight(weighted: bool, weight_mn: float, weight_mx: float) -> float:
    return random.uniform(weight_mn, weight_mx) if weighted else 1.0

def add_self_loop(graph: Graph, weighted: bool, weight_mn: float, weight_mx: float) -> Graph:
    for i in range(graph.num_nodes):
        graph.add_edge(i, i, random_weight(weighted, weight_mn, weight_mx))
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
                w = random_weight(weighted, weight_mn, weight_mx)
                graph.add_edge(node, node + 1, w)
            if node + 2 < num_nodes:
                w = random_weight(weighted, weight_mn, weight_mx)
                graph.add_edge(node, node + 2, w)
        if node % 2 == 1 and node + 2 < num_nodes:
            w = random_weight(weighted, weight_mn, weight_mx)
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
            w = random_weight(weighted, weight_mn, weight_mx)
            graph.add_edge(node, node + c, w)
        if node % c != c - 1 and node + 1 < num_nodes:
            w = random_weight(weighted, weight_mn, weight_mx)
            graph.add_edge(node, node + 1, w)

    if self_loop: 
        add_self_loop(graph, weighted, weight_mn, weight_mx)
    
    return graph

def random_tree(
    num_nodes: int, 
    seed: int | None = None, 
    weighted: bool = False,
    weight_mn: float = 0,
    weight_mx: float = 1,
    self_loop: bool = True,
) -> Graph:
    """
    Generates random tree based on Prufer sequence
    Implementation based on https://en.wikipedia.org/wiki/Pr%C3%BCfer_sequence
    """
    if seed is not None:
        random.seed(seed)

    graph = Graph(num_nodes)

    # generate prufer sequence of length num_nodes - 2
    prufer_seq = []

    for _ in range(num_nodes - 2):
        x = random.randint(0, num_nodes - 1)
        prufer_seq.append(x)

    degree = [1] * num_nodes
    for x in prufer_seq:
        degree[x] += 1

    for x in prufer_seq:
        for node in range(num_nodes):
            if degree[node] == 1:
                w = random_weight(weighted, weight_mn, weight_mx)
                graph.add_edge(node, x, w)
                degree[x] -= 1
                degree[node] -= 1
                break

    # connect the 2 remaining leaves
    leaves = [node for node in range(num_nodes) if degree[node] == 1]
    w = random_weight(weighted, weight_mn, weight_mx)
    graph.add_edge(leaves[0], leaves[1], w)

    if self_loop:
        add_self_loop(graph, weighted, weight_mn, weight_mx)

    return graph
    
def random_graph(
        num_nodes: int, 
        p: float = 0.5,
        seed: int | None = None, 
        weighted: bool = False,
        self_loop: bool = False,
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
                w = random_weight(weighted, weight_mn, weight_mx)
                graph.add_edge(i, j, w)

    if self_loop: 
        add_self_loop(graph, weighted, weight_mn, weight_mx)

    return graph

def barabisi_albert_graph(
        num_nodes: int, 
        seed: int | None = None, 
        weighted: bool = False,
        self_loop: bool = True,
        weight_mn: float = 0,
        weight_mx: float = 1,
) -> Graph:
    """
    Graph with either 4 or 5 edges to every incoming node
    """

    graph = Graph(num_nodes)

    processed = [] # keep tracks of node already in the graph

    # start with a fully connected graph
    for node in range(min(num_nodes, 5)):
        for neigh in range(node + 1, (min(num_nodes, 5))):
            w = random_weight(weighted, weight_mn, weight_mx)
            graph.add_edge(node, neigh, w)
        processed.append(node)

    for node in range(5, num_nodes):
        # add 4 or 5 edges
        edges = 4
        if random.random() < 0.5:
            edges = 5
        neighs = random.sample(processed, edges)
        for neigh in neighs:
            w = random_weight(weighted, weight_mn, weight_mx)
            graph.add_edge(node, neigh, w)

        processed.append(node)

    if self_loop:
        add_self_loop(graph, weighted, weight_mn, weight_mx)

    return graph

def community_graph(
    num_nodes: int, 
    seed: int | None = None, 
    weighted: bool = False,
    self_loop: bool = True,
    weight_mn: float = 0,
    weight_mx: float = 1,
) -> Graph:
    """
    Generates 4-Community graphs
    "first generating four disjoint Erdos-Rényi graphs with edge probability 0.7,
    followed by interconnecting their nodes with edge probability 0.01"
    """

    if seed is not None:
        random.seed(seed)

    q = num_nodes // 4
    sizes = [q, q, q, num_nodes - 3 * q]

    graph = Graph(0)
    
    for i in range(4):
        new_graph = random_graph(num_nodes=sizes[i], p=0.7, weighted=weighted,
                                 weight_mn=weight_mn, weight_mx=weight_mx, self_loop=False)
        
        offset = graph.num_nodes
        graph.merge(new_graph)
        for u in range(offset):
            for v in range(offset, graph.num_nodes):
                if random.random() < 0.05:
                    w = random_weight(weighted, weight_mn, weight_mx)
                    graph.add_edge(u, v, w)

    if self_loop:
        add_self_loop(graph, weighted, weight_mn, weight_mx)

    return graph

def caveman_graph(
    num_nodes: int, 
    seed: int | None = None, 
    weighted: bool = False,
    self_loop: bool = True,
    weight_mn: float = 0,
    weight_mx: float = 1,
) -> Graph:
    """
    4-Caveman (Watts, 1999)
    "graphs, having each of their intra-clique edges removed with probability 0.7,
    followed by inserting 0.025|V | additional shortcut edges between cliques."
    """

    if seed is not None:
        random.seed(seed)

    q = num_nodes // 4
    sizes = [q, q, q, num_nodes - 3 * q]

    graph = Graph(0)

    for i in range(4): # p = 1 because it's fully connected graph (clique)
        new_graph = random_graph(num_nodes=sizes[i], p=1.0, weighted=weighted,
                                 weight_mn=weight_mn, weight_mx=weight_mx, self_loop=False)
        
        # remove intra-clique edges with probability 0.7
        edges_to_remove = []
        for node in range(new_graph.num_nodes):
            for neigh in range(node + 1, new_graph.num_nodes):
                if random.random() < 0.7:
                    edges_to_remove.append((node, neigh))
            
        for u, v in edges_to_remove:
            new_graph.remove_edge(u, v)

        graph.merge(new_graph)
        
    num_shortcuts = int(0.025 * num_nodes)
    starts = [sum(sizes[:i]) for i in range(4)]

    for _ in range(num_shortcuts):
        c1, c2 = random.sample(range(4), 2) # pick 2 different cliques
        u = random.randint(starts[c1], starts[c1] + sizes[c1] - 1)
        v = random.randint(starts[c2], starts[c2] + sizes[c2] - 1)
        w = random_weight(weighted, weight_mn, weight_mx)
        graph.add_edge(u, v, w)

    if self_loop:
        add_self_loop(graph, weighted, weight_mn, weight_mx)

    return graph

def generate_training_graphs(
    by_category: int = 1,
    num_nodes: int = 20,
    weighted: bool = False,
    weight_mn: float = 0,
    weight_mx: float = 1,
) -> list:
    graphs = []
    for _ in range(by_category):
        graphs.extend([
            ladder_graph(num_nodes=num_nodes, weighted=weighted, weight_mn=weight_mn, weight_mx=weight_mx, self_loop=True),
            grid_graph(num_nodes=num_nodes, weighted=weighted, weight_mn=weight_mn, weight_mx=weight_mx, self_loop=True),
            random_tree(num_nodes=num_nodes, weighted=weighted, weight_mn=weight_mn, weight_mx=weight_mx, self_loop=True),
            random_graph(num_nodes=num_nodes, p=0.3, weighted=weighted, weight_mn=weight_mn, weight_mx=weight_mx, self_loop=True),
            barabisi_albert_graph(num_nodes=num_nodes, weighted=weighted, weight_mn=weight_mn, weight_mx=weight_mx, self_loop=True),
            community_graph(num_nodes=num_nodes, weighted=weighted, weight_mn=weight_mn, weight_mx=weight_mx, self_loop=True),
            caveman_graph(num_nodes=num_nodes, weighted=weighted, weight_mn=weight_mn, weight_mx=weight_mx, self_loop=True),
        ])

    return [(g, random.randrange(g.num_nodes)) for g in graphs]


if __name__ == "__main__":
    num_nodes = 5
    p = 0.5
    g = random_graph(num_nodes,
                     p=p,
                     weighted=True,
                     connected=True,
                     self_loop=True,
                     weight_mn=0,
                     weight_mx=2
    )
    print(g)

    tree = random_tree(num_nodes)
    print(tree)

    barabis_albert = barabis_albert_graph(7)
    print(barabis_albert)