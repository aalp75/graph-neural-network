import random
import copy
import time
from pathlib import Path

import torch
import torch.optim as optim
import torch.nn as nn

from graph import Graph
from model import Model
from graph_generation import generate_training_graphs
import algorithms

torch.set_float32_matmul_precision('high')

PREDEC_LAMBDA = 0.5 # used to split the loss between state and predecessors
TERM_LAMBDA = 0.2 # used to split the loss between state and termination

BCE = nn.BCEWithLogitsLoss()
MSE = nn.MSELoss()
CE = nn.CrossEntropyLoss()
# pos_weight moved to device in train function
BCE_TERM = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(5.0))  

def compute_state_loss(algo: str, start: torch.tensor, pred: torch.tensor, expected: torch.tensor):
    if algo == 'bfs':
        state_loss = BCE(pred, expected)
    elif algo in ['bf', 'cc']:
        state_loss = MSE(pred, expected)
    elif algo == 'prim':
        # predict next node
        next_node = next(j for j in range(start.shape[0]) if start[j] == 0 and expected[j] == 1)
        next_node_t = torch.tensor([next_node], dtype=torch.long, device=expected.device)

        state_loss = CE((pred.squeeze(1)).unsqueeze(0), next_node_t)

    return state_loss

def compute_reachable(algo: str, source: int, state: torch.tensor, max_dist: float) -> torch.tensor:
    n = state.shape[0]
    if algo == 'bf':
        max_dist_f = float(torch.tensor(max_dist, dtype=torch.float))
        reachable = [state[j].item() < max_dist_f and j != source for j in range(n)]
    elif algo == 'prim':
        reachable = [state[j].item() == 1 and j != source for j in range(n)]
    else:  # no predecessor tracking
        reachable = [False] * n
    return torch.tensor(reachable, device=state.device, dtype=torch.bool)

def run_algo(algo: str,
                  model: Model,
                  graph: Graph,
                  source: int,
                  samples: list,
                  max_dist: int,
                  device: torch.device,
                  optimizer: torch.optim.Optimizer | None = None, 
                  train: bool = True
) -> tuple:
    
    acc_state_loss = torch.tensor(0.0, device=device)
    acc_predec_loss = torch.tensor(0.0, device=device)
    acc_term_loss = torch.tensor(0.0, device=device)
    num_steps = 0

    edges = graph.get_edge_tensors(device)
    h = torch.zeros(graph.num_nodes, model.hidden_dim, device=device)

    if not samples:
        return {'state': 0.0, 'predec': 0.0, 'term': 0.0, 'steps': 0}

    for state, next_state, predecessors, termination in samples:

        state_pred, p_pred, h, term_pred = model(algo, edges, state, h)

        reachable = compute_reachable(algo, source, state, max_dist)

        state_loss = compute_state_loss(algo, state, state_pred, next_state)
        predec_loss = torch.tensor(0.0, device=state.device)
        if reachable.any(): # skip for some algorithms (e.g. bfs)
            predec_loss = CE(p_pred[reachable], predecessors[reachable])
        term_loss = BCE_TERM(term_pred, termination)

        acc_state_loss += state_loss
        acc_predec_loss += predec_loss
        acc_term_loss += term_loss

        num_steps += 1

    total_loss = acc_state_loss + PREDEC_LAMBDA * acc_predec_loss + TERM_LAMBDA * acc_term_loss

    if train:
        optimizer.zero_grad()
        (total_loss / num_steps).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

    return {
        'state': acc_state_loss.item(),
        'predec': acc_predec_loss.item(),
        'term': acc_term_loss.item(),
        'steps': num_steps,
    }

def run_dataset(model: Model,
             data: list,
             device: torch.device,
             optimizer: optim.Optimizer | None = None,
             train: bool = False
) -> dict:
    """
    Compute average loss on a dataset
    Args:
        train (bool): update or not the weights of the model
    """

    loss = {'state': 0.0, 'predec': 0.0, 'term': 0.0, 'steps': 0}

    if train:
        model.train()
        random.shuffle(data)
    else:
        model.eval()

    with torch.set_grad_enabled(train):
        for algo, graph, source, sample, max_dist in data:
            result = run_algo(algo, 
                                   model, 
                                   graph, 
                                   source, 
                                   sample, 
                                   max_dist, 
                                   device, 
                                   optimizer, 
                                   train=train
            )
            for k in ('state', 'predec', 'term', 'steps'):
                loss[k] += result[k]

    n = loss['steps']
    total_loss = (loss['state'] + PREDEC_LAMBDA * loss['predec'] + TERM_LAMBDA * loss['term']) / n
    return {
        'total': total_loss,
        'state': loss['state'] / n,
        'predec': loss['predec'] / n,
        'term': loss['term'] / n,
    }

def generate_data(size: int, num_nodes: int, algos: list, device: torch.device):
    data = []
    graphs = generate_training_graphs(by_category=size, num_nodes=num_nodes,
                                      weighted=True, weight_mn=0.2, weight_mx=1.0)

    for graph, source in graphs:
        graph.get_edge_tensors(device)
        for algo in algos:
            states, predecessors, inf, terminations = algorithms.compute_states(algo, graph, source)
            steps = algorithms.generate_steps(states, predecessors, terminations)

            steps = [
                (
                    torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(1),
                    torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(1),
                    torch.tensor(predecessor, dtype=torch.long, device=device) if predecessor is not None else None,
                    torch.tensor(termination, dtype=torch.float32, device=device),
                )
                for state, next_state, predecessor, termination in steps
            ]

            data.append((algo, graph, source, steps, inf))
    return data

def train(model: Model,
          train_size: int = 20,
          val_size: int = 5,
          num_nodes: int = 20,
          epochs: int = 5,
          min_epochs: int = 5,
          lr: float = 0.0005,
          patience: int = 10,
          verbose: bool = True,
          save: bool = False,
) -> None:
    """Train the model with early stopping on validation loss"""
    
    print("Generating training data...", end=' ')
    device = next(model.parameters()).device
    BCE_TERM.pos_weight = BCE_TERM.pos_weight.to(device)
    train_data = generate_data(train_size, num_nodes, model.algos, device)
    val_data = generate_data(val_size, num_nodes, model.algos, device)
    print("Generated!")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    best_val_loss = float('inf')
    patience_counter = 0

    best_params = None

    print("Start training process...")

    start_time = time.time()
    for epoch in range(epochs): 

        train_loss = run_dataset(model, train_data, device, optimizer, train=True)
        val_loss = run_dataset(model, val_data, device, None, train=False)

        if verbose:
            print(f"Epoch {epoch + 1} | "
                  f"train loss = {train_loss['total']:.2f} (state={train_loss['state']:.2f}, predec={train_loss['predec']:.2f}, term={train_loss['term']:.2f}) | "
                  f"val loss = {val_loss['total']:.2f} (state={val_loss['state']:.2f}, predec={val_loss['predec']:.2f}, term={val_loss['term']:.2f})")

        if val_loss['total'] < best_val_loss:
            best_val_loss = val_loss['total']
            best_params = copy.deepcopy(model.state_dict())
            patience_counter = 0

            if save:
                if verbose:
                    print(f"New best loss {best_val_loss:.4f} (parameters saved)")
                Path("parameters").mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), "parameters/model.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience and epoch >= min_epochs:
                print(f"Early stopping at epoch {epoch}")
                break

    if best_params is not None:
        print(f"best val loss = {best_val_loss:.4f}")
        model.load_state_dict(best_params)

    print(f"Trained in {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    algos = ['bfs', 'bf', 'prim', 'cc']
    hidden_dim = 32
    model = Model(algos, in_dim=1, hidden_dim=hidden_dim, out_dim=1).to(device) # train on GPU if available
    model = torch.compile(model) # compile to train faster

    train_size = 100 # number of graph from each category (7 category in total)
    val_size = 20
    num_nodes = 20
    epochs = 100
    min_epochs = 400
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