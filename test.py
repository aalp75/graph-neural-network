import random

from graph import Graph
from model import Model
from graph_generation import random_graph

import algorithms as algorithms

import torch
import torch.nn as nn
import torch.optim as optim

def test_bfs(model, graph: Graph, source:int) -> None:
    # TODO: implement
    pass

def test_dfs(model, graph: Graph, source:int) -> None:
    states = algorithms.compute_dfs_states(source, graph)

    criterion = torch.nn.BCEWithLogitsLoss()

    device = next(model.parameters()).device

    h = None

    with torch.no_grad():
        for step in range(len(states) - 1):
            state = states[step]
            target = states[step + 1]

            x = torch.tensor(state, dtype=torch.float32).unsqueeze(1).to(device)
            y_true = torch.tensor(target, dtype=torch.float32).unsqueeze(1).to(device)

            y, h = model('DFS', graph, x, h)

            y_pred = y

            bce = criterion(y_pred, y_true).item()

            probs = torch.sigmoid(y_pred).squeeze(1).tolist()
            visited_pred = [i for i, p in enumerate(probs) if p > 0.5]
            visited_true = [i for i, v in enumerate(target) if v == 1.0]

            h = h.detach()

            print(f"Step {step}")
            for i in range(len(state)):
                print(f"  node {i}: input={state[i]:.0f}  prob={probs[i]:.2f}  target={target[i]:.0f}")
            print(f"Visited prediction: {visited_pred}")
            print(f"Visited target:     {visited_true}")
            print("BCE:", bce)
            print()
    
def test_bf(model, graph: Graph, source:int) -> None:
    
    states, preds, inf = algorithms.compute_bf_states(source, graph)

    criterion = torch.nn.MSELoss()
    criterion_p = torch.nn.CrossEntropyLoss()

    device = next(model.parameters()).device

    h = None

    with torch.no_grad():
        for step in range(len(states) - 1):
            state = states[step]
            target = states[step + 1]
            pred = preds[step + 1]

            x = torch.tensor(state, dtype=torch.float32).unsqueeze(1).to(device)
            y_true = torch.tensor(target, dtype=torch.float32).unsqueeze(1).to(device)
            p_true = torch.tensor(pred, dtype=torch.long).to(device)

            y, h = model('BF', graph, x, h)

            y_pred = y[0]
            p_pred = torch.stack(y[1])

            mse = criterion(y_pred, y_true).item()
            ce = criterion_p(p_pred, p_true)

            p_pred = torch.stack(y[1]).argmax(dim=1).tolist()
            y_pred = y_pred.squeeze(1).tolist()
            
            h = h.detach()
            
            print(f"Step {step}")
            for i in range(len(state)):
                print(f"Input: {state[i]:.2f}, prediction: {y_pred[i]:.2f}, target: {target[i]:.2f} "
                      f"pred prediction: {p_pred[i]}, pred target: {p_true[i].item()}")
            print("MSE:", mse)
            print("Cross Entropy:", ce.item())
            print()

def test_prim(model, graph: Graph, source:int) -> None:
    states, preds = algorithms.compute_prim_states(source, graph)

    criterion = torch.nn.BCEWithLogitsLoss()
    criterion_p = torch.nn.CrossEntropyLoss()

    device = next(model.parameters()).device

    h = None

    with torch.no_grad():
        for step in range(len(states) - 1):
            state = states[step]
            target = states[step + 1]
            pred = preds[step + 1]

            x = torch.tensor(state, dtype=torch.float32).unsqueeze(1).to(device)
            y_true = torch.tensor(target, dtype=torch.float32).unsqueeze(1).to(device)
            p_true = torch.tensor(pred, dtype=torch.long).to(device)

            mst = [i for i, v in enumerate(x) if v == 1]

            y, h = model('PRIM', graph, x, h)

            y_pred = y[0]
            p_pred = torch.stack(y[1])

            mst_pred = [i for i, v in enumerate(torch.sigmoid(y_pred).squeeze(1).tolist()) if v > 0.5]

            bce = criterion(y_pred, y_true).item()
            ce = criterion_p(p_pred, p_true)

            pred_list = y_pred.squeeze(1).tolist()
            
            h = h.detach()
            
            print(f"Step {step}")
            print(f"MST Prediction: {mst_pred}\nMST Target:     {mst}")
            print("Binary Cross Entropy:", bce)
            print("Cross Entropy:", ce.item())
            print()

def test_dijkstra(model, graph: Graph, source:int) -> None:
    
    states, preds = algorithms.compute_dijkstra_states(source, graph)

    criterion = torch.nn.MSELoss()
    criterion_p = torch.nn.CrossEntropyLoss()

    device = next(model.parameters()).device

    h = None

    with torch.no_grad():
        for step in range(len(states) - 1):
            state = states[step]
            target = states[step + 1]
            pred = preds[step + 1]

            x = torch.tensor(state, dtype=torch.float32).unsqueeze(1).to(device)
            y_true = torch.tensor(target, dtype=torch.float32).unsqueeze(1).to(device)
            p_true = torch.tensor(pred, dtype=torch.long).to(device)

            y, h = model('Dijkstra', graph, x, h)

            y_pred = y[0]
            p_pred = torch.stack(y[1])

            mse = criterion(y_pred, y_true).item()
            ce = criterion_p(p_pred, p_true)

            y_pred = y_pred.squeeze(1).tolist()
            
            h = h.detach()
            
            print(f"Step {step}")
            for i in range(len(state)):
                print(f"Input: {state[i]:.2f}, prediction: {y_pred[i]:.2f}, target: {target[i]:.2f}")
            print("MSE:", mse)
            print("Cross Entropy:", ce.item())
            print()


def test():
    model = Model(1, 32, 1)
    model.load_state_dict(torch.load("parameters/model.pt", weights_only=True))
    model.eval()

    graph = random_graph(
        num_nodes=15,
        p=0.5,
        seed=None,
        self_loop=True,
        weighted=True,
        weight_mn=0.2,
        weight_mx=2.0
    )

    source = random.randrange(graph.num_nodes)

    #test_bfs(model, graph, source)
    #test_dfs(model, graph, source)
    test_bf(model, graph, source)
    #test_prim(model, graph, source)
    #test_dijkstra(model, graph, source)

if __name__ == "__main__":
    test()