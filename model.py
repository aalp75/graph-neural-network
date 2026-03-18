import torch
import torch.nn as nn

from graph import Graph


class Encoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int) -> None:
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
    def __init__(self, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.proj = nn.Linear(2 * hidden_dim, out_dim)

    def forward(self, z: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        inp = torch.cat([z, h], dim=1)
        return self.proj(inp)
    
class Predecessor(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(2 * hidden_dim + 1, 1)

    def forward(self, graph: Graph, h: torch.Tensor) -> list:
        pred_scores = []
        for node in range(graph.num_nodes):
            scores = [torch.full((1,), float('-inf'), device=h.device, dtype=h.dtype) for _ in range(graph.num_nodes)]
            for neigh, weight in graph.adj[node]:
                if neigh == node:  # skip self-loops
                    continue
                weight_tensor = torch.tensor([weight], device=h.device, dtype=h.dtype)
                scores[neigh] = self.proj(torch.cat([h[node], h[neigh], weight_tensor]))
            
            pred_scores.append(torch.cat(scores))

        return pred_scores


class Processor(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.message = nn.Linear(2 * hidden_dim + 1, hidden_dim)

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
    
class Termination(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.term = nn.Linear(2 * hidden_dim, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        h_bar = torch.mean(h, dim=0, keepdim=True).expand(h.size(0), -1)  # [n, hidden_dim]
        logits = self.term(torch.cat([h, h_bar], dim=1))  # [n, 1]
        return logits.mean()

class Model(nn.Module):
    def __init__(self, algos: list, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()

        self.algos = algos

        self.hidden_dim = hidden_dim

        self.encoders = nn.ModuleDict({a: Encoder(in_dim, hidden_dim) for a in algos})
        self.decoders = nn.ModuleDict({a: Decoder(hidden_dim, out_dim) for a in algos})
        
        self.terminations = nn.ModuleDict({a: Termination(hidden_dim) for a in algos})

        self.predecessor = Predecessor(hidden_dim)
        self.processor = Processor(hidden_dim)

    def forward(self, 
                algo: str,
                graph: Graph, 
                x: torch.Tensor, 
                h: torch.Tensor | None = None
    ) -> tuple:
        
        if algo not in self.algos:
            raise ValueError(f"Unknown algorithm: {algo}")
        
        z = self.encoders[algo](x, h)
        h = self.processor(graph, z)
        y = self.decoders[algo](z, h)
        t = self.terminations[algo](h)
        p = self.predecessor(graph, h)

        return y, p, h, t

if __name__ == "__main__":
    algos = ['BFS', 'BF', 'PRIM', 'CC']
    model = Model(algos, 1, 32, 1)
    