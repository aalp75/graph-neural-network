import random
import copy

from graph import Graph
from model import Model
from graph_generation import random_graph, generate_training_graphs
import algorithms as algorithms

import torch
import torch.nn as nn
import torch.optim as optim

LAMBDA = 0.1 # used to split the loss between state and predecessors

def evaluate_bfs(model, graph: Graph, source: int, optimizer, device, train=True):
    states = algorithms.compute_bfs_states(source, graph)
    examples = algorithms.generate_bfs_examples('BFS', graph, states)

    h = None
    total_loss = 0.0
    num_steps = 0

    criterion = nn.BCEWithLogitsLoss()

    for algo, _, state, next_state in examples:

        x = torch.tensor(state, dtype=torch.float32).unsqueeze(1).to(device)
        y_true = torch.tensor(next_state, dtype=torch.float32).unsqueeze(1).to(device)

        y_pred, h = model(algo, graph, x, h)

        loss = criterion(y_pred, y_true)

        if train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        h = h.detach()

        total_loss += loss.item()
        num_steps += 1

    return total_loss, num_steps

def evaluate_dfs(model, graph: Graph, source: int, optimizer, device, train=True):
    states = algorithms.compute_dfs_states(source, graph)
    examples = algorithms.generate_dfs_examples('DFS', graph, states)

    h = None
    total_loss = 0.0
    num_steps = 0

    criterion = nn.BCEWithLogitsLoss()

    for algo, _, state, next_state in examples:

        x = torch.tensor(state, dtype=torch.float32).unsqueeze(1).to(device)
        y_true = torch.tensor(next_state, dtype=torch.float32).unsqueeze(1).to(device)

        y_pred, h = model(algo, graph, x, h)

        loss = criterion(y_pred, y_true)

        if train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        h = h.detach()

        total_loss += loss.item()
        num_steps += 1

    return total_loss, num_steps

def evaluate_bf(model, graph: Graph, source: int, optimizer, device, train=True) -> None:
    states, preds, inf = algorithms.compute_bf_states(source, graph)
    examples = algorithms.generate_bf_examples('BF', graph, states, preds)

    criterion = nn.MSELoss()
    criterion_p = nn.CrossEntropyLoss()

    h = None
    total_loss = 0.0
    num_steps = 0

    for algo, _, state, next_state, pred in examples:

        x = torch.tensor(state, dtype=torch.float32).unsqueeze(1).to(device)
        y_true = torch.tensor(next_state, dtype=torch.float32).unsqueeze(1).to(device)
        p_true = torch.tensor(pred, dtype=torch.long).to(device)
        reachable = [i for i in range(graph.num_nodes) if state[i] < inf and i != source]

        y, h = model(algo, graph, x, h)

        y_pred = y[0]
        p_pred = torch.stack(y[1])

        if reachable:
            idx = torch.tensor(reachable, device=device)
            loss_pred = criterion_p(p_pred[idx], p_true[idx])
        else:
            loss_pred = 0.0

        loss = criterion(y_pred, y_true) + LAMBDA * loss_pred

        if train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        h = h.detach()

        total_loss += loss.item()
        num_steps += 1

    return total_loss, num_steps

def evaluate_prim(model, graph: Graph, source: int, optimizer, device, train=True) -> None:
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

        y, h = model(algo, graph, x, h)

        y_pred = y[0]
        p_pred = torch.stack(y[1])

        if reachable:
            idx = torch.tensor(reachable, device=device)
            loss_pred = criterion_p(p_pred[idx], p_true[idx])
        else:
            loss_pred = 0.0

        loss = criterion(y_pred, y_true) + LAMBDA * loss_pred

        if train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        h = h.detach()

        total_loss += loss.item()
        num_steps += 1

    return total_loss, num_steps

def evaluate_dijkstra(model, graph: Graph, source: int, optimizer, device, train=True) -> None:
    states, preds, inf = algorithms.compute_dijkstra_states(source, graph)
    examples = algorithms.generate_dijkstra_examples('Dijkstra', graph, states, preds)

    criterion = nn.MSELoss()
    criterion_p = nn.CrossEntropyLoss()

    h = None
    total_loss = 0.0
    num_steps = 0

    for algo, _, state, next_state, pred in examples:

        x = torch.tensor(state, dtype=torch.float32).unsqueeze(1).to(device)
        y_true = torch.tensor(next_state, dtype=torch.float32).unsqueeze(1).to(device)
        p_true = torch.tensor(pred, dtype=torch.long).to(device)
        reachable = [i for i in range(graph.num_nodes) if state[i] < inf and i != source]

        y, h = model(algo, graph, x, h)

        y_pred = y[0]
        p_pred = torch.stack(y[1])

        if reachable:
            idx = torch.tensor(reachable, device=device)
            loss_pred = criterion_p(p_pred[idx], p_true[idx])
        else:
            loss_pred = 0.0

        loss = criterion(y_pred, y_true) + LAMBDA * loss_pred

        if train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        h = h.detach()

        total_loss += loss.item()
        num_steps += 1

    return total_loss, num_steps

def evaluate(model, data, optimizer, device):
    """
    Compute average loss on a dataset without updating weights
    """

    total_loss = 0.0
    total_steps = 0

    model.eval()
    with torch.no_grad():
        for graph, source in data:

            # evaluate BFS
            loss_bfs, steps_bfs = evaluate_bfs(model, graph, source, optimizer, device, train=False)
            total_loss += loss_bfs
            total_steps += steps_bfs

            # evaluate DFS
            loss_dfs, steps_dfs = evaluate_dfs(model, graph, source, optimizer, device, train=False)
            total_loss += loss_dfs
            total_steps += steps_dfs
            
            # evaluate BF
            loss_bf, steps_bf = evaluate_bf(model, graph, source, optimizer, device, train=False)
            total_loss += loss_bf
            total_steps += steps_bf

            # evaluate Prim
            loss_prim, steps_prim = evaluate_prim(model, graph, source, optimizer, device, train=False)
            total_loss += loss_prim
            total_steps += steps_prim

            # evaluate Dijkstra
            loss_dijkstra, steps_dijkstra = evaluate_dijkstra(model, graph, source, optimizer, device, train=False)
            total_loss += loss_dijkstra
            total_steps += steps_dijkstra

    return total_loss / total_steps

def train(model, # torch model
          train_size: int,
          val_size: int,
          num_nodes: int, 
          num_epochs: int, 
          lr:float = 0.0005,
          patience:int = 10,
          verbose:bool = True, 
          save:bool = False,
):
    """
    train
    """
    
    print("Generating training data...", end=' ')
    train_data = generate_training_graphs(by_category=train_size, num_nodes=num_nodes,
                                          weighted=True, weight_mn=0.2, weight_mx=2.0)
    
    val_data =  generate_training_graphs(by_category=val_size, num_nodes=num_nodes,
                                          weighted=True, weight_mn=0.2, weight_mx=2.0)
    
    print("Generated!")
    
    device = next(model.parameters()).device

    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_loss = float('inf')
    patience_counter = 0

    best_params = None

    print("Start training process...")

    for epoch in range(num_epochs):

        model.train()

        total_loss = 0.
        total_steps = 0
    
        for graph, source in train_data:
            
            # train BFS
            loss_bfs, steps_bfs = evaluate_bfs(model, graph, source, optimizer, device, train=True)
            total_loss += loss_bfs
            total_steps += steps_bfs

            # train DFS
            loss_dfs, steps_dfs = evaluate_dfs(model, graph, source, optimizer, device, train=True)
            total_loss += loss_dfs
            total_steps += steps_dfs
            
            # train BF
            loss_bf, steps_bf = evaluate_bf(model, graph, source, optimizer, device, train=True)
            total_loss += loss_bf
            total_steps += steps_bf

            # train Prim
            loss_prim, steps_prim = evaluate_prim(model, graph, source, optimizer, device, train=True)
            total_loss += loss_prim
            total_steps += steps_prim

            # train Dijkstra
            loss_dijkstra, steps_dijkstra = evaluate_dijkstra(model, graph, source, optimizer, device, train=True)
            total_loss += loss_dijkstra
            total_steps += steps_dijkstra

        current_val_loss = evaluate(model, val_data, optimizer, device)

        if verbose:
            print(f"Epoch {epoch} | train loss = {total_loss / total_steps:.4f} | val loss = {current_val_loss:.4f}")

        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            best_params = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    if best_params is not None:
        model.load_state_dict(best_params)
    if save:
        torch.save(model.state_dict(), "parameters/model.pt")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    model = Model(1, 32, 1).to(device) # train on GPU if available
    train_size = 20 # number of graph of each category (7 category in total)
    val_size = 2
    num_nodes = 10
    num_epochs = 100
    train(model, train_size, val_size, num_nodes, num_epochs, verbose=True, save=True)