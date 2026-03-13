import torch
import torch.nn as nn

from graph import Graph

class EncoderBFS(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden_dim)

    def forward(self, x):
        return self.proj(x)

class DecoderBFS(nn.Module):
    def __init__(self, hidden_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, h):
        return self.proj(h)

class EncoderBF(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden_dim)

    def forward(self, x):
        return self.proj(x)

class DecoderBF(nn.Module):
    def __init__(self, hidden_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, h):
        return self.proj(h)
    
class Processor(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()

        self.message = nn.Sequential(
            nn.Linear(2 * hidden_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.update = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, g, h):
        n = g.num_nodes
        device = h.device
        dtype = h.dtype
        d = h.size(1)

        incoming = [[] for _ in range(n)]

        for u in range(n):
            for v, w in g.adj[u]:
                weight_tensor = torch.tensor([w], device=device, dtype=dtype)
                msg_input = torch.cat([h[u], h[v], weight_tensor], dim=0)
                msg = self.message(msg_input)
                incoming[v].append(msg)

        agg_rows = []
        for v in range(n):
            if incoming[v]:
                agg_v = torch.stack(incoming[v], dim=0).max(dim=0).values
            else:
                agg_v = torch.zeros(d, device=device, dtype=dtype)
            agg_rows.append(agg_v)

        agg = torch.stack(agg_rows, dim=0)

        h = self.update(torch.cat([h, agg], dim=1))
        return h
                        

class Model(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()

        # encoder
        self.encoder_bfs = EncoderBFS(in_dim, hidden_dim)
        self.decoder_bfs = DecoderBFS(hidden_dim, out_dim)
        
        # decoder
        self.encoder_bf = EncoderBF(in_dim, hidden_dim)
        self.decoder_bf = DecoderBF(hidden_dim, out_dim)

        # shared processor
        self.processor = Processor(hidden_dim)

    def forward(self, algo: str, g: Graph, x):
        if algo == 'BFS':
            h = self.encoder_bfs(x)
        elif algo == 'BF':
            h = self.encoder_bf(x)

        h = self.processor(g, h)

        if algo == 'BFS':
            y = self.decoder_bfs(h)
        elif algo == 'BF':
            y = self.decoder_bf(h)

        return y
    
if __name__ == "__main__":
    model = Model(1, 32, 1)