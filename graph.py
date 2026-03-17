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

    def remove_edge(self, u:int , v:int) -> None:
        """
        Remove edge (u, v)
        Complexity is O(degree) due to list filtering
        """
        self.adj[u] = [(n, w) for n, w in self.adj[u] if n != v]
        if u != v:
            self.adj[v] = [(n, w) for n, w in self.adj[v] if n != u]

    def __repr__(self) -> str:
        res = f"Graph with {self.num_nodes} nodes and adjacency list:\n"
        for node in range(self.num_nodes):
            edges = [(v, round(w, 3)) for v, w in self.adj[node]]
            res += f"  Node {node}: {edges}\n"
        return res
    
    def merge(self, other: 'Graph') -> None:
        """
        Merge 2 graphs: self and other
        """
        offset = self.num_nodes
        self.num_nodes += other.num_nodes
        self.adj.extend([[] for _ in range(other.num_nodes)])
        for node in range(other.num_nodes):
            for neigh, weight in other.adj[node]:
                self.adj[node + offset].append((neigh + offset, weight))
        
if __name__ == "__main__":
    graph = Graph(3)
    graph.add_edge(0, 1, 5.3)
    graph.add_edge(0, 2, 2.5)
    print(graph)


    