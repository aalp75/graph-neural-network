import torch

class Graph:
    """Graph is represented as an adjacency list [neighbour, weight]"""
    def __init__(self, num_nodes: int, adjacency_list: list | None = None) -> None:
        self.num_nodes = num_nodes
        if adjacency_list is None:
            self.adj = [[] for _ in range(num_nodes)]
        else:
            self.adj = adjacency_list

        # cache attributes
        self.longest_path = None
        self.edge_tensors_dict = dict() # edge_tensors by devices and dtype

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
        res = res = f"Graph {self.num_nodes} nodes\n"
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

    def get_longest_path(self) -> None:
        if self.longest_path is None:
            self.compute_longest_path()
        return self.longest_path
    
    def build_edge_tensors(self, key: str, device: torch.device) -> None:
        sources = []
        dests = []
        weights = []

        for j in range(self.num_nodes):
            for i, w in self.adj[j]:
                sources.append(j)
                dests.append(i)
                weights.append(w)

        if len(sources) == 0:
            sources_t = torch.empty(0, dtype=torch.long, device=device)
            dests_t = torch.empty(0, dtype=torch.long, device=device)
            weights_t = torch.empty((0, 1), dtype=torch.float32, device=device)
        else:
            sources_t = torch.tensor(sources, dtype=torch.long, device=device)
            dests_t = torch.tensor(dests, dtype=torch.long, device=device)
            weights_t = torch.tensor(weights, dtype=torch.float32, device=device).unsqueeze(1)

        self.edge_tensors_dict[key] = (sources_t, dests_t, weights_t)

    def get_edge_tensors(self, device: torch.device) -> tuple:
        key = str(device)
        if key not in self.edge_tensors_dict:
            self.build_edge_tensors(key, device)
        return self.edge_tensors_dict[key]
        
if __name__ == "__main__":
    graph = Graph(3)
    graph.add_edge(0, 1, 5.3)
    graph.add_edge(0, 2, 2.5)
    print(graph)
    print(graph.get_longest_path())


    