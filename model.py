import torch
import torch.nn as nn

from graph import Graph

class Encoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.proj = nn.Linear(in_dim + hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        #if h is None:
            #h = torch.zeros(x.size(0), self.hidden_dim, device=x.device, dtype=x.dtype)
        return self.proj(torch.cat([x, h], dim=1))

class Decoder(nn.Module):
    def __init__(self, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(2 * hidden_dim, out_dim)

    def forward(self, z: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        return self.proj(torch.cat([z, h], dim=1))

class Processor(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

        # M (message) and U (update) are linear projections
        self.message = nn.Linear(2 * hidden_dim + 1, hidden_dim)
        self.update  = nn.Linear(2 * hidden_dim, hidden_dim)

    def forward(self, edges: tuple, z: torch.Tensor) -> torch.Tensor:
        n = z.size(0)
        device = z.device
        dtype = z.dtype

        sources, dests, weights = edges

        # no edge
        if sources.numel() == 0:
            m = torch.zeros((n, self.hidden_dim), device=device, dtype=dtype)
            h = self.update(torch.cat([z, m], dim=1))
            return h

        inp = torch.cat([z[dests], z[sources], weights], dim=1)   # (E, 2h+1)
        messages = self.message(inp)

        # max-aggregation
        m = torch.full((n, self.hidden_dim), float('-inf'), device=device, dtype=dtype)

        m.scatter_reduce_(
            0, # dim
            dests.unsqueeze(1).expand_as(messages),
            messages,
            reduce="amax",
            include_self=True
        )
        # replace -inf by 0
        m = torch.where(torch.isneginf(m), torch.zeros_like(m), m)

        # update
        h = self.update(torch.cat([z, m], dim=1))
        return h
    
class Termination(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_dim, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        h_mean = h.mean(dim=0, keepdim=True)
        return self.proj(h_mean).squeeze()
    
class Predecessor(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(2 * hidden_dim + 1, 1)

    def forward(self, edges: tuple, h: torch.Tensor) -> torch.Tensor:
        n = h.size(0)
        device = h.device
        dtype = h.dtype

        sources, dests, weights = edges

        sources = sources.to(device=device)
        dests = dests.to(device=device)
        weights = weights.to(device=device, dtype=dtype)

        scores = torch.full((n, n), -1e9, device=h.device, dtype=h.dtype)
        edge_input = torch.cat([h[dests], h[sources], weights], dim=1)
        scores[dests, sources] = self.proj(edge_input).squeeze(1)

        return scores

class Model(nn.Module):
    def __init__(self, algos: list, in_dim: int = 1, hidden_dim: int = 32, out_dim: int = 1) -> None:
        super().__init__()

        self.algos = algos
        self.hidden_dim = hidden_dim

        self.encoders = nn.ModuleDict({algo: Encoder(in_dim, hidden_dim) for algo in algos})
        self.decoders = nn.ModuleDict({algo: Decoder(hidden_dim, out_dim) for algo in algos})
        self.terminations = nn.ModuleDict({algo: Termination(hidden_dim) for algo in algos})

        self.processor = Processor(hidden_dim)

        self.predecessor = Predecessor(hidden_dim)

    def forward(self, algo: str, edges: tuple, x: torch.Tensor, h: torch.Tensor) -> tuple:

        if algo not in self.algos:
            raise ValueError(f"Unknown algorithm: {algo}")

        z = self.encoders[algo](x, h) 
        h_new = self.processor(edges, z)
        y = self.decoders[algo](z, h_new)
        t = self.terminations[algo](h_new)
        p = self.predecessor(edges, h_new)

        return y, p, h_new, t

if __name__ == "__main__":
    algos = ['bfs', 'bf', 'prim', 'cc']
    model = Model(algos, 1, 32, 1)
