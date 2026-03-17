import torch
import torch.nn as nn

from graph import Graph


class Encoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.proj = nn.Linear(in_dim + hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor, h: torch.Tensor | None = None) -> torch.Tensor:
        if h is None:
            n = x.size(0)
            h = torch.zeros(n, self.hidden_dim, device=x.device, dtype=x.dtype)

        inp = torch.cat([x, h], dim=1)
        return self.proj(inp)

class Decoder(nn.Module):
    def __init__(self, hidden_dim: int, out_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.proj = nn.Linear(2 * hidden_dim, out_dim)

    def forward(self, z: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        inp = torch.cat([z, h], dim=1)
        return self.proj(inp)
    
class Predecessor(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.proj = nn.Linear(2 * hidden_dim + 1, 1)

    def forward(self, h:torch.Tensor, graph:Graph):
        pred_scores = []
        for node in range(graph.num_nodes):
            scores = [torch.zeros(1, device=h.device, dtype=h.dtype) for _ in range(graph.num_nodes)]
            for neigh, weight in graph.adj[node]:
                if neigh == node:  # skip self-loops
                    continue
                weight_tensor = torch.tensor([weight], device=h.device, dtype=h.dtype)
                scores[neigh] = self.proj(torch.cat([h[node], h[neigh], weight_tensor]))
            
            pred_scores.append(torch.cat(scores))

        return pred_scores


class Processor(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.message = nn.Sequential(
            nn.Linear(2 * hidden_dim + 1, hidden_dim),
        )

        self.update = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, graph: Graph, z: torch.Tensor) -> torch.Tensor:
        n = graph.num_nodes
        device = z.device
        dtype = z.dtype

        h = [None] * n

        for u in range(n):
            all_messages = []
            for v, weight in graph.adj[u]:
                weight_tensor = torch.tensor([weight], device=device, dtype=dtype)
                msg_input = torch.cat([z[u], z[v], weight_tensor], dim=0)
                msg = self.message(msg_input)
                all_messages.append(msg)

            if all_messages:
                aggregated_msg = torch.stack(all_messages, dim=0).max(dim=0).values
            else:
                aggregated_msg = torch.zeros_like(z[u])

            h[u] = self.update(torch.cat([z[u], aggregated_msg], dim=0))

        return torch.stack(h, dim=0)


class Model(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()

        self.hidden_dim = hidden_dim

        self.encoder_bfs = Encoder(in_dim, hidden_dim)
        self.decoder_bfs = Decoder(hidden_dim, out_dim)

        self.encoder_dfs = Encoder(in_dim, hidden_dim)
        self.decoder_dfs = Decoder(hidden_dim, out_dim)

        self.encoder_bf = Encoder(in_dim, hidden_dim)
        self.decoder_bf = Decoder(hidden_dim, out_dim)

        self.encoder_prim = Encoder(in_dim, hidden_dim)
        self.decoder_prim = Decoder(hidden_dim, out_dim)

        self.encoder_dijkstra = Encoder(in_dim, hidden_dim)
        self.decoder_dijkstra = Decoder(hidden_dim, out_dim)

        self.predecessor = Predecessor(hidden_dim)

        self.processor = Processor(hidden_dim)

    def forward(self, 
                algo: str,
                graph: Graph, 
                x: torch.Tensor, 
                h: torch.Tensor | None = None
    ):
        if algo == 'BFS':
            z = self.encoder_bfs(x, h)
        elif algo == 'DFS':
            z = self.encoder_dfs(x, h)
        elif algo == 'BF':
            z = self.encoder_bf(x, h)
        elif algo == 'PRIM':
            z = self.encoder_prim(x, h)
        elif algo == 'Dijkstra':
            z = self.encoder_dijkstra(x, h)
        else:
            raise ValueError(f"Unknown algorithm: {algo}")

        h = self.processor(graph, z)

        if algo == 'BFS':
            y = self.decoder_bfs(z, h)
        elif algo == 'DFS':
            y = self.decoder_dfs(z, h)
        elif algo == 'BF':
            y = self.decoder_bf(z, h)
            pred_scores = self.predecessor(h, graph)
            y = [y, pred_scores]
        elif algo == 'PRIM':
            y = self.decoder_prim(z, h)
            pred_scores = self.predecessor(h, graph)
            y = [y, pred_scores]
        elif algo == 'Dijkstra':
            y = self.decoder_dijkstra(z, h)
            pred_scores = self.predecessor(h, graph)
            y = [y, pred_scores]

        return y, h

if __name__ == "__main__":
    model = Model(1, 32, 1)
    