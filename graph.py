class Graph:
    def __init__(self, num_nodes: int, adjacency_list: list) -> None:
        self.num_nodes = num_nodes
        self.adj = adjacency_list

    def add_edge(self, u: int, v: int) -> None:
        self.adj[u].append(v)
        self.adj[v].append(u)

    def __repr__(self):
        return f"Graph(num_nodes={self.num_nodes}, adj={self.adj})"
    