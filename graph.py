class Graph:
    """
    Graph is represented as a adjacency list [neighbour, weight]
    """
    def __init__(self, num_nodes: int, adjacency_list: list) -> None:
        self.num_nodes = num_nodes
        self.adj = adjacency_list

    def add_edge(self, u: int, v: int, w: float = 1.0) -> None:
        self.adj[u].append((v, w))
        self.adj[v].append((u, w))

    def __repr__(self) -> str:
        return f"Graph(num_nodes={self.num_nodes}, adj={self.adj})"
    