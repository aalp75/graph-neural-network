import random

from graph import Graph
from model import Model
from graph_generation import graph_generation
from algoritms import compute_bfs_states, compute_bellman_ford_states, generate_examples

import torch
import torch.nn as nn
import torch.optim as optim

def test():
    model = Model(1, 32, 1)
    model.load_state_dict(torch.load("parameters/model.pt"))
    model.eval()

    graph = graph_generation(
        num_nodes=20,
        p=0.2,
        seed=None,
        self_loop=True,
        weighted=True,
        weight_mn=0.2,
        weight_mx=2.0
    )

    source = random.randrange(graph.num_nodes)
    states = compute_bellman_ford_states(source, graph)

    criterion = torch.nn.MSELoss()

    device = next(model.parameters()).device

    h = None

    with torch.no_grad():
        for step in range(len(states) - 1):
            state = states[step]
            target = states[step + 1]

            x = torch.tensor(state, dtype=torch.float32).unsqueeze(1).to(device)
            y_true = torch.tensor(target, dtype=torch.float32).unsqueeze(1).to(device)

            y_pred, h = model('BF', graph, x, h)
            mse = criterion(y_pred, y_true).item()

            pred_list = y_pred.squeeze(1).tolist()
            h.detach()
            
            print(f"Step {step}")
            for i in range(len(state)):
                print(f"Input: {state[i]}, prediction: {pred_list[i]:.2f} target: {target[i]}")
            print("MSE:", mse)
            print()

if __name__ == "__main__":
    test()