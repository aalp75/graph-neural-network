import torch
import torch.nn as nn

from graph import Graph


class Encoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.proj = nn.Linear(in_dim + hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
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

    def forward(self, graph: Graph, h: torch.Tensor) -> torch.Tensor:
        n = graph.num_nodes
        sources, dists, weights = graph.get_edges_tensor(h.device, h.dtype)

        scores = torch.full((n, n), float('-inf'), device=h.device, dtype=h.dtype)
        edge_input = torch.cat([h[sources], h[dists], weights], dim=1)
        scores[sources, dists] = self.proj(edge_input).squeeze(1)

        return scores

class Processor(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.message = nn.Linear(2 * hidden_dim + 1, hidden_dim)

        #self.update = nn.Sequential(
        #    nn.Linear(2 * hidden_dim, hidden_dim),
        #    nn.ReLU(),
        #    nn.Linear(hidden_dim, hidden_dim)
        #)
        self.update = nn.Linear(2 * hidden_dim, hidden_dim)

    def forward(self, graph: Graph, z: torch.Tensor) -> torch.Tensor:
        n = graph.num_nodes
        sources, dists, weights = graph.get_edges_tensor(z.device, z.dtype)

        if sources.numel() > 0:
            msg_input = torch.cat([z[sources], z[dists], weights], dim=1)
            messages  = self.message(msg_input)

            agg = torch.full((n, messages.size(1)), float('-inf'), device=z.device, dtype=z.dtype)
            agg.scatter_reduce_(0, dists.unsqueeze(1).expand_as(messages), messages, reduce='amax', include_self=True)
            agg = agg.masked_fill(agg.isinf(), 0.0)
        else:
            agg = torch.zeros(n, self.message.out_features, device=z.device, dtype=z.dtype)

        return self.update(torch.cat([z, agg], dim=1))
    
class Termination(nn.Module):
    def __init__(self, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.term = nn.Sequential(
            nn.Linear(2 * hidden_dim + 2 * out_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, h: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        h_mean = h.mean(dim=0, keepdim=True)
        h_max = h.max(dim=0, keepdim=True).values
        y_mean = y.mean(dim=0, keepdim=True)
        y_max = y.max(dim=0, keepdim=True).values
        pooled = torch.cat([h_mean, h_max, y_mean, y_max], dim=1)
        return self.term(pooled).squeeze()


class Model(nn.Module):
    def __init__(self, algos: list, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()

        self.algos = algos

        self.hidden_dim = hidden_dim

        self.encoders = nn.ModuleDict({a: Encoder(in_dim, hidden_dim) for a in algos})
        self.decoders = nn.ModuleDict({a: Decoder(hidden_dim, out_dim) for a in algos})

        self.processor = Processor(hidden_dim)

        self.terminations = nn.ModuleDict({a: Termination(hidden_dim, out_dim) for a in algos})
        
        self.predecessor = Predecessor(hidden_dim)

    def forward(self, 
                algo: str,
                graph: Graph, 
                x: torch.Tensor, 
                h: torch.Tensor | None = None
    ) -> tuple:
        
        if algo not in self.algos:
            raise ValueError(f"Unknown algorithm: {algo}")
        
        z = self.encoders[algo](x, h)
        new_h = self.processor(graph, z)
        y = self.decoders[algo](z, new_h)
        t = self.terminations[algo](new_h, y)
        p = self.predecessor(graph, new_h)

        return y, p, new_h, t

if __name__ == "__main__":
    algos = ['BFS', 'BF', 'PRIM', 'CC']
    model = Model(algos, 1, 32, 1)
    