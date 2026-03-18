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

LAMBDA = 0.2 # used to split the loss between state and predecessors

BCE = nn.BCEWithLogitsLoss()
MSE = nn.MSELoss()
CE = nn.CrossEntropyLoss()


def termination_bce(term_pred: torch.Tensor, term_true: torch.Tensor, num_steps: int) -> torch.Tensor:
    """
    Balance between 0 and 1 for the termination prediction because the 0 appears 
    more than 1
    """
    pos_weight = torch.tensor(float(num_steps - 1), device=term_pred.device)
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)(term_pred, term_true)

def compute_loss(algo: str,
                 y_pred: torch.Tensor, y_true: torch.Tensor,
                 p_pred: torch.Tensor | None, p_true: torch.Tensor | None,
                 term_pred: torch.Tensor, term_true: torch.Tensor,
                 reachable: list,
                 num_steps: int,
                 device: torch.device,
                 next_node: torch.Tensor | None = None,
                 state: list | None = None,
) -> torch.Tensor:
    match algo:
        case 'BFS':
            state_loss = BCE(y_pred, y_true)
        case 'BF' | 'CC':
            state_loss = MSE(y_pred, y_true)
        case 'PRIM':
            mask = torch.tensor([float('-inf') if state[j] == 1 else 0.0 for j in range(len(state))], device=y_pred.device, dtype=y_pred.dtype)
            state_loss = CE((y_pred.squeeze(1) + mask).unsqueeze(0), next_node)

    parent_loss = 0.0
    if reachable:
        idx = torch.tensor(reachable, device=device)
        parent_loss = LAMBDA * CE(p_pred[idx], p_true[idx])

    return state_loss + termination_bce(term_pred, term_true, num_steps) + parent_loss

def evaluate_algo(algo: str, model: Model, graph: Graph, source: int, device: torch.device, optimizer: optim.Optimizer | None = None, train: bool = True) -> tuple:
    states, parents, inf, _ = algorithms.compute_states(algo, graph, source)
    examples = algorithms.generate_examples(states, parents)

    h = None
    total_loss = 0.0
    accumulated_loss = None
    num_steps = 0

    for i, (state, next_state, parent) in enumerate(examples):

        x = torch.tensor(state, dtype=torch.float32).unsqueeze(1).to(device)
        y_true = torch.tensor(next_state, dtype=torch.float32).unsqueeze(1).to(device)
        p_true = None
        if parent is not None:
            p_true = torch.tensor(parent, dtype=torch.long).to(device)
        term_true = torch.tensor(1.0 if i == len(examples) - 1 else 0.0, device=device)

        next_node = None
        match algo:
            case 'BF':
                reachable = [node for node in range(graph.num_nodes) if next_state[node] < inf and node != source]
            case 'PRIM':
                reachable = [node for node in range(graph.num_nodes) if next_state[node] == 1 and node != source]
                next_node = torch.tensor(
                    [next(j for j in range(graph.num_nodes) if state[j] == 0 and next_state[j] == 1)],
                    dtype=torch.long, device=device
                )
            case _:
                reachable = []

        y_pred, p_pred, h, term_pred = model(algo, graph, x, h)
        p_pred = torch.stack(p_pred)

        loss = compute_loss(algo, y_pred, y_true, p_pred, p_true, term_pred, term_true, reachable, len(examples), device, next_node, state)

        accumulated_loss = loss if accumulated_loss is None else accumulated_loss + loss
        h = h.detach()
        total_loss += loss.item()
        num_steps += 1

    if train and accumulated_loss is not None:
        optimizer.zero_grad()
        accumulated_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    return total_loss, num_steps

def evaluate(model: Model, data: list, device: torch.device, optimizer: optim.Optimizer | None = None, train: bool = False) -> float:
    """
    Compute average loss on a dataset without updating weights
    """

    total_loss = 0.0
    total_steps = 0

    if train:
        model.train()
        random.shuffle(data)
    else:
        model.eval()

    with torch.set_grad_enabled(train):
        for graph, source in data:
            for algo in model.algos:
                loss, steps = evaluate_algo(algo, model, graph, source, device, optimizer, train=train)
                total_loss  += loss
                total_steps += steps

    return total_loss / total_steps

def train(model: Model,
          train_size: int,
          val_size: int,
          num_nodes: int, 
          num_epochs: int, 
          lr:float = 0.0005,
          patience:int = 10,
          verbose:bool = True, 
          save: bool = False,
) -> None:
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

        train_loss = evaluate(model, train_data, device, optimizer, train=True)
        val_loss = evaluate(model, val_data, device, None, train=False)

        if verbose:
            print(f"Epoch {epoch} | train loss = {train_loss:.4f} | val loss = {val_loss:.4f}")

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
    algos = ['BFS', 'BF', 'PRIM', 'CC']
    model = Model(algos, 1, 32, 1).to(device) # train on GPU if available
    train_size = 10 # number of graph of each category (7 category in total)
    val_size = 5
    num_nodes = 10
    num_epochs = 50
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