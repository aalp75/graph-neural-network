import torch

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

        self.longest_path = None
        self.edges_tensor = None # used for the neural network input

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

    def compute_longest_path(self) -> float:
        """
        Compute longest shortest path using Floyd-Warshall
        Time complexity is O(graph.num_nodes ^ 3)
        """
        n =self.num_nodes
        dist = [[float('inf')] * n for _ in range(n)]

        for u in range(n):
            dist[u][u] = 0
            for v, w in self.adj[u]:
                dist[u][v] = min(dist[u][v], w)

        for k in range(n):
            for u in range(n):
                for v in range(n):
                    dist[u][v] = min(dist[u][v], dist[u][k] + dist[k][v])

        self.longest_path = max(dist[u][v] for u in range(n) for v in range(n) if dist[u][v] < float('inf'))

    def get_longest_path(self) -> float:
        if self.longest_path is None:
            self.compute_longest_path()
        return self.longest_path
    
    def edges_to_tensor(self, device: torch.device = None, dtype: torch.dtype = None) -> None:
        """
        Convert edges to tensors and cache them to speed up neural network computation
        """
        sources, dists, weights = [], [], []
        for node in range(self.num_nodes):
            for neigh, weight in self.adj[node]:
                sources.append(node)
                dists.append(neigh)
                weights.append(weight)
        self.edges_tensor = (
            torch.tensor(sources, device=device),
            torch.tensor(dists, device=device),
            torch.tensor(weights, device=device, dtype=dtype).unsqueeze(1),
        )

    def get_edges_tensor(self, device, dtype) -> tuple:
        if self.edges_tensor is None:
            self.edges_to_tensor(device, dtype)
        return self.edges_tensor
        
if __name__ == "__main__":
    graph = Graph(3)
    graph.add_edge(0, 1, 5.3)
    graph.add_edge(0, 2, 2.5)
    print(graph)
    print(graph.get_longest_path())


    