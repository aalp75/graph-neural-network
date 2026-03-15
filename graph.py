class Graph:
    """
    Graph is represented as an adjacency list [neighbour, weight]
    """
    def __init__(self, num_nodes: int, adjacency_list: list | None = None) -> None:
        self.num_nodes = num_nodes
        if adjacency_list is None:
            self.adj = [[] for _ in range(num_nodes)]
        else:
            self.adj = adjacency_list

    def add_edge(self, u: int, v: int, w: float = 1.0) -> None:
        self.adj[u].append((v, w))
        if u != v:
            self.adj[v].append((u, w))

    def __repr__(self) -> str:
        return f"Graph(num_nodes={self.num_nodes}, adj={self.adj})"
    
if __name__ == "__main__":
    graph = Graph(3)
    graph.add_edge(0, 1, 5.3)
    graph.add_edge(0, 2, 2.5)
    print(graph)


    