import random

from graph import Graph
from model import Model
from graph_generation import graph_generation

import algorithms as algorithms

import torch
import torch.nn as nn
import torch.optim as optim

def test_bfs(model, graph: Graph, source:int) -> None:
    # TODO: implement
    pass

def test_bf(model, graph: Graph, source:int) -> None:
    
    states, preds = algorithms.compute_bf_states(source, graph)

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

            y_pred = y_pred.squeeze(1).tolist()
            p_pred_idx = torch.softmax(p_pred, dim=1).argmax(dim=1).tolist()
            
            h = h.detach()
            
            print(f"Step {step}")
            for i in range(len(state)):
                print(f"Input: {state[i]:.2f}, prediction: {y_pred[i]:.2f}, target: {target[i]:.2f}, "
                      f"pred predecessor: {p_pred_idx[i]}, true predecessor: {pred[i]}")
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

            p_pred_idx = torch.softmax(p_pred, dim=1).argmax(dim=1).tolist()

            mst_pred = [i for i, v in enumerate(torch.sigmoid(y_pred).squeeze(1).tolist()) if v > 0.5]

            bce = criterion(y_pred, y_true).item()
            ce = criterion_p(p_pred, p_true)

            pred_list = y_pred.squeeze(1).tolist()
            p_pred_idx = torch.softmax(p_pred, dim=1).argmax(dim=1).tolist()
            
            h = h.detach()
            
            print(f"Step {step}")
            print(f"MST Prediction: {mst_pred}\nMST Target: {mst}")
            print(f"pred predecessor: {p_pred_idx},\ntrue predecessor: {pred}")
            print("Binary Cross Entropy:", bce)
            print("Cross Entropy:", ce.item())
            print()



def test():
    model = Model(1, 32, 1)
    model.load_state_dict(torch.load("parameters/model.pt"))
    model.eval()

    graph = graph_generation(
        num_nodes=15,
        p=0.3,
        seed=None,
        self_loop=True,
        weighted=True,
        weight_mn=0.2,
        weight_mx=2.0
    )

    source = random.randrange(graph.num_nodes)

    #test_bfs(model, graph, source)
    test_bf(model, graph, source)
    test_prim(model, graph, source)

if __name__ == "__main__":
    test()