import random
import copy
from pathlib import Path

from graph import Graph
from model import Model
from graph_generation import generate_training_graphs
import algorithms as algorithms

import torch
import torch.nn as nn
import torch.optim as optim

PRED_LAMBDA = 0.2 # used to split the loss between state and predecessors
TERM_LAMBDA = 0.05 # used to split the loss between state and termination

BCE = nn.BCEWithLogitsLoss()
MSE = nn.MSELoss()
CE = nn.CrossEntropyLoss()

def termination_bce(term_pred: torch.Tensor, term_true: torch.Tensor, num_steps: int) -> torch.Tensor:
    """
    Weighted BCE for termination prediction: 0 appears more than 1
    """
    #print(f"term prediction {torch.sigmoid(term_pred).item()}: term target: {term_true}")
    pos_weight = torch.tensor([float(num_steps - 1)], device=term_pred.device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    return criterion(term_pred.view(1), term_true.view(1))

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
            n = y_pred.shape[0]
            state_loss = MSE(y_pred / n, y_true / n)
        case 'PRIM':
            mask = torch.tensor([float('-inf') if state[j] == 1 else 0.0 for j in range(len(state))], device=y_pred.device, dtype=y_pred.dtype)
            state_loss = CE((y_pred.squeeze(1) + mask).unsqueeze(0), next_node)

    parent_loss = torch.tensor(0.0, device=device)
    if reachable:
        idx = torch.tensor(reachable, device=device)
        parent_loss = PRED_LAMBDA * CE(p_pred[idx], p_true[idx])

    #print("state loss:", state_loss)
    #print("termination loss:", termination_bce(term_pred, term_true, num_steps))
    #print("parent loss: ", parent_loss)
    return state_loss, parent_loss, termination_bce(term_pred, term_true, num_steps)

def evaluate_algo(algo: str, 
                  model: Model, 
                  graph: Graph, 
                  source: int, 
                  device: torch.device, 
                  optimizer: optim.Optimizer | None = None, 
                  train: bool = True
) -> tuple:
    states, parents, inf, _ = algorithms.compute_states(algo, graph, source)
    examples = algorithms.generate_examples(states, parents)

    h = torch.zeros(graph.num_nodes, model.hidden_dim, device=device)
    total_state_loss = 0.0
    total_parent_loss = 0.0
    total_term_loss = 0.0
    accumulated_state_loss = None
    accumulated_parent_loss = None
    accumulated_term_loss = None
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

        state_loss, parent_loss, term_loss = compute_loss(algo, y_pred, y_true, p_pred, p_true, term_pred, term_true, reachable, len(examples), device, next_node, state)

        accumulated_state_loss = state_loss if accumulated_state_loss is None else accumulated_state_loss + state_loss
        accumulated_parent_loss = parent_loss if accumulated_parent_loss is None else accumulated_parent_loss + parent_loss
        accumulated_term_loss = term_loss if accumulated_term_loss is None else accumulated_term_loss + term_loss
        h = h.detach()
        total_state_loss += state_loss.item()
        total_parent_loss += parent_loss.item()
        total_term_loss += term_loss.item()
        num_steps += 1

    if train and accumulated_state_loss is not None:
        optimizer.zero_grad()
        (accumulated_state_loss / num_steps + accumulated_parent_loss / num_steps + TERM_LAMBDA * accumulated_term_loss / num_steps).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

    return total_state_loss, total_parent_loss, total_term_loss, num_steps

def evaluate(model: Model, data: list, device: torch.device, optimizer: optim.Optimizer | None = None, train: bool = False) -> float:
    """
    Compute average loss on a dataset
    inputs:
        train: update the weights of the model
    """

    total_loss = 0.0
    total_state_loss = 0.0
    total_parent_loss = 0.0
    total_term_loss = 0.0
    total_steps = 0

    if train:
        model.train()
        random.shuffle(data)
    else:
        model.eval()

    with torch.set_grad_enabled(train):
        for graph, source in data:
            for algo in model.algos:
                state_loss, parent_loss, term_loss, steps = evaluate_algo(algo, model, graph, source, device, optimizer, train=train)
                total_state_loss  += state_loss
                total_parent_loss += parent_loss
                total_term_loss   += term_loss
                total_loss        += state_loss + parent_loss + TERM_LAMBDA * term_loss
                total_steps       += steps

    return total_loss / total_steps, total_state_loss / total_steps, total_parent_loss / total_steps, total_term_loss / total_steps

def train(model: Model,
          train_size: int = 20,
          val_size: int = 5,
          num_nodes: int = 20,
          epochs: int = 5,
          min_epochs: int = 5,
          lr:float = 0.0005,
          patience:int = 10,
          verbose:bool = True, 
          save: bool = False,
) -> None:
    """
    Train the model with early stopping on validation loss"
    """
    
    print("Generating training data...", end=' ')
    train_data = generate_training_graphs(by_category=train_size, num_nodes=num_nodes,
                                          weighted=True, weight_mn=0.2, weight_mx=2.0)
    
    val_data =  generate_training_graphs(by_category=val_size, num_nodes=num_nodes,
                                          weighted=True, weight_mn=0.2, weight_mx=2.0)
    
    device = next(model.parameters()).device
    
    for graph, source in train_data + val_data:
        graph.get_edges_tensor(device, torch.float32)
    
    print("Generated!")

    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_loss = float('inf')
    patience_counter = 0

    best_params = None

    print("Start training process...")

    # TODO: precompute all stats and cache it

    for epoch in range(epochs):

        train_loss, train_state_loss, train_parent_loss, train_term_loss = evaluate(model, train_data, device, optimizer, train=True)
        val_loss, val_state_loss, val_parent_loss, val_term_loss = evaluate(model, val_data, device, None, train=False)

        if verbose:
            print(f"Epoch {epoch} | train loss = {train_loss:.4f} (state={train_state_loss:.4f}, parent={train_parent_loss:.4f}, term={train_term_loss:.4f}) | val loss = {val_loss:.4f} (state={val_state_loss:.4f}, parent={val_parent_loss:.4f}, term={val_term_loss:.4f})")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_params = copy.deepcopy(model.state_dict())
            patience_counter = 0

            if save:
                Path("parameters").mkdir(parents=True, exist_ok=True)
                print(f"New best loss {best_val_loss}, parameters saved!")
                torch.save(model.state_dict(), "parameters/model.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience and epoch >= min_epochs:
                print(f"Early stopping at epoch {epoch}")
                break

    if best_params is not None:
        print(f"best val loss = {best_val_loss:.4f}")
        model.load_state_dict(best_params)
    if save:
        Path("parameters").mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), "parameters/model.pt")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    algos = ['BFS', 'BF', 'PRIM', 'CC']
    model = Model(algos, 1, 32, 1).to(device) # train on GPU if available
    torch.set_float32_matmul_precision('high')
    #model = torch.compile(model)
    train_size = 100 # number of graph of each category (7 category in total)
    val_size = 20
    num_nodes = 20
    epochs = 400
    min_epochs = 300
    patience = 50
    train(model,
          train_size=train_size,
          val_size=val_size, 
          num_nodes=num_nodes, 
          epochs=epochs,
          min_epochs=min_epochs,
          lr=5e-4,
          patience=patience,
          verbose=True,
          save=True
    )