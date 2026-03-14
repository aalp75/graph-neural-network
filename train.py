import random

from graph import Graph
from model import Model
from graph_generation import graph_generation
from algoritms import compute_bfs_states, compute_bellman_ford_states, generate_examples

import torch
import torch.nn as nn
import torch.optim as optim


def train_bfs(model, graph: Graph, optimizer, criterion, device) -> None:
    source = random.randrange(graph.num_nodes)
    states = compute_bfs_states(source, graph)
    examples = generate_examples('BFS', graph, states)

    h = None
    total_loss = 0.0
    num_steps = 0

    for algo, _, state, next_state in examples:

        x = torch.tensor(state, dtype=torch.float32).unsqueeze(1).to(device)
        y_true = torch.tensor(next_state, dtype=torch.float32).unsqueeze(1).to(device)

        optimizer.zero_grad()

        y_pred, h = model(algo, graph, x, h)

        loss = criterion(y_pred, y_true)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        h = h.detach()

        total_loss += loss.item()
        num_steps += 1

    return total_loss, num_steps

def train_bf(model, graph: Graph, optimizer, criterion, device) -> None:
    source = random.randrange(graph.num_nodes)
    states = compute_bellman_ford_states(source, graph)
    examples = generate_examples('BF', graph, states)

    h = None
    total_loss = 0.0
    num_steps = 0

    for algo, _, state, next_state in examples:

        x = torch.tensor(state, dtype=torch.float32).unsqueeze(1).to(device)
        y_true = torch.tensor(next_state, dtype=torch.float32).unsqueeze(1).to(device)

        optimizer.zero_grad()

        y_pred, h = model(algo, graph, x, h)

        loss = criterion(y_pred, y_true)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        h = h.detach()

        total_loss += loss.item()
        num_steps += 1

    return total_loss, num_steps

def train(model, num_graphs, num_nodes, num_epochs: int, verbose=True, save=False):
    # generate graphs
    
    graphs = []
    for _ in range(num_graphs):

        graph = graph_generation(
            num_nodes=num_nodes,
            p=0.4,
            weighted=True,
            seed=None,
            self_loop=True,
            weight_mn=0.0,
            weight_mx=2.0
        )

        graphs.append(graph)

    device = next(model.parameters()).device

    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    criterion_bfs = nn.BCEWithLogitsLoss()
    criterion_bf = nn.MSELoss()

    for epoch in range(num_epochs):

        total_loss = 0.
        total_steps = 0
    
        for graph in graphs:
            
            # train BFS
            loss_bfs, steps_bfs = train_bfs(model, graph, optimizer, criterion_bfs, device)
            total_loss += loss_bfs
            total_steps += steps_bfs
            
            # train BF
            loss_bf, steps_bf = train_bf(model, graph, optimizer, criterion_bf, device)
            total_loss += loss_bf
            total_steps += steps_bf

        if verbose:
            print(f"Epoch {epoch} loss = {total_loss / total_steps:.4f}")

    if save:
        torch.save(model.state_dict(), "parameters/model.pt")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    model = Model(1, 32, 1).to(device) # train on GPU if available
    num_graphs = 50
    num_nodes = 20
    num_epochs = 10
    train(model, num_graphs, num_nodes, num_epochs, verbose=True, save=True)
