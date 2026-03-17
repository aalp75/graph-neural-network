import random
import copy
from pathlib import Path

from graph import Graph
from model import Model
from graph_generation import random_graph, generate_training_graphs
import algorithms as algorithms

import torch
import torch.nn as nn
import torch.optim as optim

LAMBDA = 0.1 # used to split the loss between state and predecessors

BCE = nn.BCEWithLogitsLoss()
MSE = nn.MSELoss()
CE = nn.CrossEntropyLoss()

def termination_bce(term_pred, term_true, num_steps):
    """
    Balance between 0 and 1 for the termination prediction because the 0 appears 
    more than 1
    """
    pos_weight = torch.tensor(float(num_steps - 1), device=term_pred.device)
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)(term_pred, term_true)

def compute_loss(algo, y_pred, y_true, p_pred, p_true, term_pred, term_true, reachable, num_steps, device):
    match algo:
        case 'BFS' | 'DFS':
            state_loss = BCE(y_pred, y_true)
        case 'BF' | 'Dijkstra':
            state_loss = MSE(y_pred, y_true)
        case 'PRIM':
            state_loss = BCE(y_pred, y_true)

    pred_loss = 0.0
    if reachable:
        idx = torch.tensor(reachable, device=device)
        pred_loss = LAMBDA * CE(p_pred[idx], p_true[idx])

    return state_loss + termination_bce(term_pred, term_true, num_steps) + pred_loss

def evaluate_algo(algo, model, graph: Graph, source: int, optimizer, device, train=True):
    states, preds, inf = algorithms.compute_states(algo, graph, source)
    examples = algorithms.generate_examples(algo, graph, states, preds)

    h = None
    total_loss = 0.0
    num_steps = 0

    for i, (_, _, state, next_state, parent) in enumerate(examples):

        x      = torch.tensor(state,      dtype=torch.float32).unsqueeze(1).to(device)
        y_true = torch.tensor(next_state, dtype=torch.float32).unsqueeze(1).to(device)
        p_true = torch.tensor(parent,     dtype=torch.long).to(device)
        term_true = torch.tensor(1.0 if i == len(examples) - 1 else 0.0, device=device)

        match algo:
            case 'BF' | 'Dijkstra':
                reachable = [node for node in range(graph.num_nodes) if next_state[node] < inf and node != source]
            case 'PRIM':
                reachable = [node for node in range(graph.num_nodes) if next_state[node] == 1 and node != source]
            case _:
                reachable = []

        y_pred, p_pred, h, term_pred = model(algo, graph, x, h)
        p_pred = torch.stack(p_pred)

        loss = compute_loss(algo, y_pred, y_true, p_pred, p_true, term_pred, term_true, reachable, len(examples), device)

        if train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        h = h.detach()
        total_loss += loss.item()
        num_steps += 1

    return total_loss, num_steps

def evaluate_bfs(model, graph: Graph, source: int, optimizer, device, train=True):
    states = algorithms.compute_bfs_states(source, graph)
    examples = algorithms.generate_examples('BFS', graph, states)

    h = None
    total_loss = 0.0
    num_steps = 0

    criterion = nn.BCEWithLogitsLoss()

    for i, (algo, _, state, next_state, parent) in enumerate(examples):

        x = torch.tensor(state, dtype=torch.float32).unsqueeze(1).to(device)

        y_true = torch.tensor(next_state, dtype=torch.float32).unsqueeze(1).to(device)
        term_true = torch.tensor(1.0 if i == len(examples) - 1 else 0.0, device=device)

        y_pred, p_pred, h, term_pred = model(algo, graph, x, h)

        loss = criterion(y_pred, y_true) + termination_bce(term_pred, term_true, len(examples))

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
    examples = algorithms.generate_examples('BF', graph, states, preds)

    h = None
    total_loss = 0.0
    num_steps = 0

    for i, (algo, _, state, next_state, parent) in enumerate(examples):

        x = torch.tensor(state, dtype=torch.float32).unsqueeze(1).to(device)

        y_true = torch.tensor(next_state, dtype=torch.float32).unsqueeze(1).to(device)
        p_true = torch.tensor(parent, dtype=torch.long).to(device)
        term_true = torch.tensor(1.0 if i == len(examples) - 1 else 0.0, device=device)

        reachable = [node for node in range(graph.num_nodes) if next_state[node] < inf and node != source]

        y_pred, p_pred, h, term_pred = model(algo, graph, x, h)

        p_pred = torch.stack(p_pred)

        if reachable:
            idx = torch.tensor(reachable, device=device)
            loss_pred = CE(p_pred[idx], p_true[idx])
        else:
            loss_pred = 0.0

        loss = MSE(y_pred, y_true) + termination_bce(term_pred, term_true, len(examples)) + LAMBDA * loss_pred

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
    examples = algorithms.generate_examples('PRIM', graph, states, preds)

    criterion = nn.BCEWithLogitsLoss()
    criterion_p = nn.CrossEntropyLoss()

    h = None
    total_loss = 0.0
    num_steps = 0

    for i, (algo, _, state, next_state, parent) in enumerate(examples):

        x = torch.tensor(state, dtype=torch.float32).unsqueeze(1).to(device)

        y_true = torch.tensor(next_state, dtype=torch.float32).unsqueeze(1).to(device)
        p_true = torch.tensor(parent, dtype=torch.long).to(device)
        term_true = torch.tensor(1.0 if i == len(examples) - 1 else 0.0, device=device)

        reachable = [i for i in range(graph.num_nodes) if state[i] == 1 and i != source] # in mst

        y_pred, p_pred, h, term_pred = model(algo, graph, x, h)

        p_pred = torch.stack(p_pred)

        if reachable:
            idx = torch.tensor(reachable, device=device)
            loss_pred = criterion_p(p_pred[idx], p_true[idx])
        else:
            loss_pred = 0.0

        loss = criterion(y_pred, y_true) + LAMBDA * loss_pred + termination_bce(term_pred, term_true, len(examples))

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

    for i, (algo, _, state, next_state) in enumerate(examples):

        x = torch.tensor(state, dtype=torch.float32).unsqueeze(1).to(device)

        y_true = torch.tensor(next_state, dtype=torch.float32).unsqueeze(1).to(device)
        term_true = torch.tensor(1.0 if i == len(examples) - 1 else 0.0, device=device)

        y_pred, h, term_pred = model(algo, graph, x, h)

        loss = criterion(y_pred, y_true) + termination_bce(term_pred, term_true, len(examples))

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

    for i, (algo, _, state, next_state, pred) in enumerate(examples):

        x = torch.tensor(state, dtype=torch.float32).unsqueeze(1).to(device)

        y_true = torch.tensor(next_state, dtype=torch.float32).unsqueeze(1).to(device)
        p_true = torch.tensor(pred, dtype=torch.long).to(device)
        term_true = torch.tensor(1.0 if i == len(examples) - 1 else 0.0, device=device)

        reachable = [node for node in range(graph.num_nodes) if next_state[node] < inf and node != source]

        y, h, term_pred = model(algo, graph, x, h)

        y_pred = y[0]
        p_pred = torch.stack(y[1])

        if reachable:
            idx = torch.tensor(reachable, device=device)
            loss_pred = criterion_p(p_pred[idx], p_true[idx])
        else:
            loss_pred = 0.0

        loss = criterion(y_pred, y_true) + LAMBDA * loss_pred + termination_bce(term_pred, term_true, len(examples))

        if train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        h = h.detach()

        total_loss += loss.item()
        num_steps += 1

    return total_loss, num_steps

def evaluate(model, data, optimizer, device, train=False):
    """
    Compute average loss on a dataset without updating weights
    """

    total_loss = 0.0
    total_steps = 0

    if train:
        model.train()
    else:
        model.eval()

    ctx = torch.no_grad() if not train else torch.enable_grad()
    with ctx:
        for graph, source in data:
            for algo in model.algos:
                loss, steps = evaluate_algo(algo, model, graph, source, optimizer, device, train=train)
                total_loss  += loss
                total_steps += steps

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
    train function
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
    
        train_loss = evaluate(model, train_data, optimizer, device, train=True)
        val_loss = evaluate(model, val_data, optimizer, device, train=False)

        if verbose:
            print(f"Epoch {epoch} | train loss = {total_loss / total_steps:.4f} | val loss = {current_val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
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
        Path("parameters").mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), "parameters/model.pt")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    algos = ['BFS', 'BF', 'PRIM']
    model = Model(algos, 1, 32, 1).to(device) # train on GPU if available
    train_size = 20 # number of graph of each category (7 category in total)
    val_size = 5
    num_nodes = 10
    num_epochs = 30
    patience = 10
    train(model, 
          train_size, 
          val_size, 
          num_nodes, 
          num_epochs, 
          lr=0.0005,
          patience=10, 
          verbose=True, 
          save=True
    )