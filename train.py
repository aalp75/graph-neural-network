import random

from graph import Graph
from model import Model
from graph_generation import graph_generation
import algorithms as algorithms

import torch
import torch.nn as nn
import torch.optim as optim


def train_bfs(model, graph: Graph, source: int, optimizer, criterion, device) -> None:
    states = algorithms.compute_bfs_states(source, graph)
    examples = algorithms.generate_bfs_examples('BFS', graph, states)

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

def train_bf(model, graph: Graph, source: int, optimizer, criterion, device) -> None:
    states, preds = algorithms.compute_bf_states(source, graph)
    examples = algorithms.generate_bf_examples('BF', graph, states, preds)

    criterion_p = nn.CrossEntropyLoss()

    h = None
    total_loss = 0.0
    num_steps = 0

    for algo, _, state, next_state, pred in examples:

        x = torch.tensor(state, dtype=torch.float32).unsqueeze(1).to(device)
        y_true = torch.tensor(next_state, dtype=torch.float32).unsqueeze(1).to(device)
        p_true = torch.tensor(pred, dtype=torch.long).to(device)
        reachable = [i for i in range(graph.num_nodes) if state[i] < graph.num_nodes and i != source]

        optimizer.zero_grad()

        y, h = model(algo, graph, x, h)

        y_pred = y[0]
        p_pred = torch.stack(y[1])

        if reachable:
            idx = torch.tensor(reachable, device=device)
            loss_pred = criterion_p(p_pred[idx], p_true[idx])
        else:
            loss_pred = 0.0

        loss = criterion(y_pred, y_true) + loss_pred / graph.num_nodes

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        h = h.detach()

        total_loss += loss.item()
        num_steps += 1

    return total_loss, num_steps

def train_prim(model, graph: Graph, source: int, optimizer, device) -> None:
    states, preds = algorithms.compute_prim_states(source, graph)
    examples = algorithms.generate_prim_examples('PRIM', graph, states, preds)

    criterion = nn.BCEWithLogitsLoss()
    criterion_p = nn.CrossEntropyLoss()

    h = None
    total_loss = 0.0
    num_steps = 0

    for algo, _, state, next_state, pred in examples:

        x = torch.tensor(state, dtype=torch.float32).unsqueeze(1).to(device)
        y_true = torch.tensor(next_state, dtype=torch.float32).unsqueeze(1).to(device)
        p_true = torch.tensor(pred, dtype=torch.long).to(device)
        reachable = [i for i in range(graph.num_nodes) if state[i] == 1 and i != source] # in mst

        optimizer.zero_grad()

        y, h = model(algo, graph, x, h)

        y_pred = y[0]
        p_pred = torch.stack(y[1])

        if reachable:
            idx = torch.tensor(reachable, device=device)
            loss_pred = criterion_p(p_pred[idx], p_true[idx])
        else:
            loss_pred = 0.0

        loss = criterion(y_pred, y_true) + loss_pred

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        h = h.detach()

        total_loss += loss.item()
        num_steps += 1

    return total_loss, num_steps

def train(model, # torch model
          train_size: int,
          val_size: int,
          num_nodes: int, 
          num_epochs: int, 
          lr:float = 0.0005,
          verbose:bool = True, 
          save:bool = False,
):
    """
    train
    """
    
    train_data = []
    for _ in range(train_size):

        graph = graph_generation(
            num_nodes=num_nodes,
            p=0.4,
            weighted=True,
            seed=None,
            self_loop=True,
            weight_mn=0.2,
            weight_mx=2.0
        )
        source = random.randrange(graph.num_nodes)
        train_data.append((graph, source))

    val_data = []
    for _ in range(val_size):
        graph = graph_generation(
            num_nodes=num_nodes,
            p=0.4,
            weighted=True,
            seed=None,
            self_loop=True,
            weight_mn=0.2,
            weight_mx=2.0
        )
        source = random.randrange(graph.num_nodes)
        val_data.append((graph, source))

    device = next(model.parameters()).device

    optimizer = optim.Adam(model.parameters(), lr=lr)

    criterion_bfs = nn.BCEWithLogitsLoss()
    criterion_bf = nn.MSELoss()

    for epoch in range(num_epochs):

        total_loss = 0.
        total_steps = 0
    
        for graph, source in train_data:
            
            # train BFS
            loss_bfs, steps_bfs = train_bfs(model, graph, source, optimizer, criterion_bfs, device)
            total_loss += loss_bfs
            total_steps += steps_bfs
            
            # train BF
            loss_bf, steps_bf = train_bf(model, graph, source, optimizer, criterion_bf, device)
            total_loss += loss_bf
            total_steps += steps_bf

            # train Prim
            loss_prim, steps_prim = train_prim(model, graph, source, optimizer, device)
            total_loss += loss_prim
            total_steps += steps_prim

        if verbose:
            print(f"Epoch {epoch} loss = {total_loss / total_steps:.4f}")

    if save:
        torch.save(model.state_dict(), "parameters/model.pt")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    model = Model(1, 32, 1).to(device) # train on GPU if available
    train_size = 50
    val_size = 5
    num_nodes = 15
    num_epochs = 10
    train(model, train_size, val_size, num_nodes, num_epochs, verbose=True, save=True)
