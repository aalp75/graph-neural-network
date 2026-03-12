import random
import torch
import torch.nn as nn
from graph import Graph

class GNN(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, g, state: torch.Tensor) -> torch.Tensor:

        if state.dim() == 1:
            state = state.unsqueeze(1)

        n = g.num_nodes
        neighbor_sum = torch.zeros((n, 1), dtype=state.dtype)

        for node in range(n):
            for neigh in g.adj[node]:
                neighbor_sum[node, 0] += state[neigh, 0]

        x = torch.cat([state, neighbor_sum], dim=1)

        return self.linear(x)

    def train_model(
        self,
        dataset,
        epochs: int = 200,
        lr: float = 0.01
    ):

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.BCEWithLogitsLoss()

        for epoch in range(epochs):

            total_loss = 0.0

            for g, state, target in dataset:

                state = torch.tensor(state, dtype=torch.float32)
                target = torch.tensor(target, dtype=torch.float32).unsqueeze(1)

                optimizer.zero_grad()

                logits = self(g, state)

                loss = loss_fn(logits, target)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            if epoch % 20 == 0:
                print(f"Epoch {epoch} | loss = {total_loss/len(dataset):.6f}")

    def predict_next_state(
        self,
        g,
        state,
        threshold: float = 0.5
    ):
        self.eval()

        state = torch.tensor(state, dtype=torch.float32)

        with torch.no_grad():

            logits = self(g, state)

            probs = torch.sigmoid(logits)

            preds = (probs >= threshold).float()

        return probs.squeeze(1), preds.squeeze(1)