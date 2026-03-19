import math
import random

from model import Model
from graph_generation import random_graph, generate_training_graphs

import algorithms as algorithms

import torch
import torch.nn as nn


def test_bfs(model: Model, data: list, device: torch.device, details: bool = False) -> None:
    print("** TESTING BFS **")
    model.eval()

    total_score = 0
    total_last_step_score = 0
    total_steps = 0
    term_score = 0

    for graph, source in data:

        states = algorithms.compute_bfs_states(graph, source)

        if len(states) < 2:
            continue

        h = torch.zeros(graph.num_nodes, model.hidden_dim, device=device)

        if details:
            print('State 0 = ', states[0])

        x = torch.tensor(states[0], dtype=torch.float32).unsqueeze(1).to(device)

        with torch.no_grad():
            for step in range(len(states) - 1):
                target = states[step + 1]
                y_true = torch.tensor(target, dtype=torch.float32).unsqueeze(1).to(device)
                term_true = torch.tensor([1 if step == (len(states) - 2) else 0])

                y_pred, p_pred, h, term_pred = model('BFS', graph, x, h)
                term_prob = torch.sigmoid(term_pred).item()

                probs = torch.sigmoid(y_pred).squeeze(1).tolist()
                prediction = [1 if p > 0.5 else 0 for p in probs]

                visited_pred = [i for i, p in enumerate(probs) if p > 0.5]
                visited_true = [i for i, v in enumerate(target) if v == 1.0]

                score = sum(p == t for p, t in zip(prediction, target)) / len(target)

                h = h.detach()

                if details:
                    bce = nn.BCEWithLogitsLoss()(y_pred, y_true).item()
                    print(f"Step {step}")
                    print(f"Visited prediction: {visited_pred}")
                    print(f"Visited target:     {visited_true}")
                    print(f"Termination: prob={term_prob:.2f}"
                          f" ({'yes' if term_prob > 0.5 else 'no'}), "
                          f"target={'yes' if term_true.item() else 'no'}")
                    print("BCE:", bce)
                    print(f"{score * 100:.2f}% predicted")
                    print()

                total_score += score
                total_steps += 1
                term_score += (term_true == (term_prob > 0.5))

                x = torch.sigmoid(y_pred).detach()

                if term_prob > 0.5:
                    break

            target = states[-1]
            score = sum(p == t for p, t in zip(prediction, target)) / len(target)
            total_last_step_score += score

    print(f"Total score: {total_score / total_steps * 100:.2f}%")
    print(f"Last step score: {total_last_step_score / len(data) * 100:.2f}%")
    print(f"Termination score: {term_score.item() / total_steps * 100:.2f}%")

def test_bf(model: Model, data: list, device: torch.device, details: bool = False) -> None:
    print("** TESTING BELLMAN-FORD **")
    model.eval()

    total_mse = 0.0
    total_p_loss = 0.0
    total_last_step_mse = 0.0
    total_last_step_p_acc = 0.0
    total_steps = 0
    term_score = 0

    for graph, source in data:
        states, parents, inf = algorithms.compute_bf_states(graph, source)

        if len(states) < 2:
            continue

        h = torch.zeros(graph.num_nodes, model.hidden_dim, device=device)
        x = torch.tensor(states[0], dtype=torch.float32).unsqueeze(1).to(device)

        with torch.no_grad():
            for step in range(len(states) - 1):
                target = states[step + 1]
                parent = parents[step + 1]

                reachable = [i for i in range(graph.num_nodes) if i != source and target[i] < inf]

                y_true = torch.tensor(target, dtype=torch.float32).unsqueeze(1).to(device)
                p_true = torch.tensor(parent, dtype=torch.long).to(device)
                term_true = torch.tensor([1 if step == (len(states) - 2) else 0])

                y_pred, p_pred, h, term_pred = model('BF', graph, x, h)
                term_prob = torch.sigmoid(term_pred).item()

                mse = nn.MSELoss()(y_pred, y_true).item()
                idx = torch.tensor(reachable, device=device)
                ce = nn.CrossEntropyLoss()(p_pred[idx], p_true[idx]).item() if reachable else 0.0
                p_pred_list = p_pred.argmax(dim=1).tolist()

                pred_acc = sum(p_pred_list[i] == parent[i] for i in reachable) / len(reachable) if reachable else 1.0

                h = h.detach()

                prediction = [round(x, 2) for x in y_pred.squeeze(1).tolist()]
                target = [round(x, 2) for x in target]

                if details:
                    print(f"Step {step}")
                    print(f"Prediction: {prediction}")
                    print(f"Target:     {target}")
                    print(f"MSE: {mse:.4f}  Predecessor CE: {ce:.4f}  Pred acc: {pred_acc*100:.1f}%")
                    #print(f"Termination: prob={term_prob:.2f}"
                    #      f" ({'yes' if term_prob > 0.5 else 'no'}), "
                    #      f"target={'yes' if term_true.item() else 'no'}")
                    print()

                total_mse += mse
                total_p_loss += pred_acc
                total_steps += 1
                term_score += (term_true == (term_prob > 0.5))

                x = y_true

                if term_prob > 0.5:
                    break

            final_target = states[-1]
            final_parent = parents[-1]
            total_last_step_mse += nn.MSELoss()(y_pred, torch.tensor(final_target, dtype=torch.float32).unsqueeze(1).to(device)).item()
            final_reachable = [i for i in range(graph.num_nodes) if i != source and final_target[i] < inf]
            final_p_pred = p_pred.argmax(dim=1).tolist()
            total_last_step_p_acc += sum(final_p_pred[i] == final_parent[i] for i in final_reachable) / len(final_reachable) if final_reachable else 1.0

    print(f"Total mean MSE: {total_mse / total_steps:.4f}")
    print(f"Total mean predecessor acc: {total_p_loss / total_steps * 100:.2f}%")
    print(f"Last step mean MSE: {total_last_step_mse / len(data):.4f}")
    print(f"Last step predecessor acc: {total_last_step_p_acc / len(data) * 100:.2f}%")
    print(f"Termination score: {term_score.item() / total_steps * 100:.2f}%")

def test_prim(model: Model, data: list, device: torch.device, details: bool = False) -> None:
    print("** TESTING PRIM **")
    model.eval()

    total_ce = 0.0
    total_next_node_acc = 0.0
    total_p_acc = 0.0
    total_last_step_p_acc = 0.0
    total_steps = 0
    term_score = 0

    for graph, source in data:
        states, parents = algorithms.compute_prim_states(graph, source)

        if details:
            print(f"Source = {source}")

        if len(states) < 2:
            continue

        h = torch.zeros(graph.num_nodes, model.hidden_dim, device=device)
        x = torch.tensor(states[0], dtype=torch.float32).unsqueeze(1).to(device)

        with torch.no_grad():
            for step in range(len(states) - 1):
                state = states[step]
                target = states[step + 1]
                parent = parents[step + 1]
                next_node = next(i for i in range(graph.num_nodes) if state[i] == 0 and target[i] == 1)

                next_node_true = torch.tensor([next_node], dtype=torch.long).to(device)
                p_true = torch.tensor(parent, dtype=torch.long).to(device)
                term_true = torch.tensor([1 if step == (len(states) - 2) else 0])

                y_pred, p_pred, h, term_pred = model('PRIM', graph, x, h)
                term_prob = torch.sigmoid(term_pred).item()

                ce = nn.CrossEntropyLoss()(y_pred.squeeze(1).unsqueeze(0), next_node_true).item()
                p_ce = nn.CrossEntropyLoss()(p_pred, p_true).item()
                logits = y_pred.squeeze(1)
                undiscovered = [i for i in range(graph.num_nodes) if state[i] == 0]
                next_node_pred = max(undiscovered, key=lambda i: logits[i].item())
                p_pred_list = p_pred.argmax(dim=1).tolist()

                reachable = [i for i in range(graph.num_nodes) if target[i] == 1 and i != source]
                p_acc = sum(p_pred_list[i] == parent[i] for i in reachable) / len(reachable) if reachable else 1.0

                h = h.detach()

                if details:
                    mst_pred = [i for i, v in enumerate(state) if v == 1] + [next_node_pred]
                    mst_true = [i for i, v in enumerate(target) if v == 1]
                    print(f"Step {step}")
                    print(f"MST prediction: {mst_pred}")
                    print(f"MST target:     {mst_true}")
                    print(f"Next node: pred={next_node_pred}  target={next_node}")
                    print(f"Next node CE: {ce:.4f}  Predecessor CE: {p_ce:.4f}  Pred acc: {p_acc*100:.1f}%")
                    print(f"Termination: prob={term_prob:.2f}"
                          f" ({'yes' if term_prob > 0.5 else 'no'}), "
                          f"target={'yes' if term_true.item() else 'no'}")
                    print()

                total_ce += ce
                total_next_node_acc += (next_node_pred == next_node)
                total_p_acc += p_acc
                total_steps += 1
                term_score += (term_true == (term_prob > 0.5))

                x = x.clone()
                x[next_node_pred] = 1.0

                if term_prob > 0.5:
                    break

            final_parent = parents[-1]
            final_reachable = [i for i in range(graph.num_nodes) if states[-1][i] == 1 and i != source]
            final_p_pred = p_pred.argmax(dim=1).tolist()
            total_last_step_p_acc += sum(final_p_pred[i] == final_parent[i] for i in final_reachable) / len(final_reachable) if final_reachable else 1.0

    print(f"Total mean next node CE: {total_ce / total_steps:.4f}")
    print(f"Total mean next node acc: {total_next_node_acc / total_steps * 100:.2f}%")
    print(f"Total mean predecessor acc: {total_p_acc / total_steps * 100:.2f}%")
    print(f"Last step predecessor acc: {total_last_step_p_acc / len(data) * 100:.2f}%")
    print(f"Termination score: {term_score.item() / total_steps * 100:.2f}%")

def test_cc(model: Model, data: list, device: torch.device, details: bool = False) -> None:
    print("** TESTING CONNECTED COMPONENTS **")
    model.eval()

    total_acc = 0.0
    total_last_step_acc = 0.0
    total_steps = 0
    term_score = 0

    for graph, _ in data:
        states = algorithms.compute_cc_states(graph)

        if len(states) < 2:
            continue

        h = torch.zeros(graph.num_nodes, model.hidden_dim, device=device)
        x = torch.tensor(states[0], dtype=torch.float32).unsqueeze(1).to(device)

        with torch.no_grad():
            for step in range(len(states) - 1):
                state = states[step]
                target = states[step + 1]

                term_true = torch.tensor([1 if step == (len(states) - 2) else 0])

                y_pred, _, h, term_pred = model('CC', graph, x, h)
                term_prob = torch.sigmoid(term_pred).item()

                h = h.detach()

                comp_pred = [round(v) for v in y_pred.squeeze(1).tolist()]
                comp_true = [round(v) for v in target]
                acc = sum(p == t for p, t in zip(comp_pred, comp_true)) / len(comp_true)

                if details:
                    print(f"Step {step}")
                    print(f"Prediction: {comp_pred}")
                    print(f"Target:     {comp_true}")
                    print(f"Accuracy: {acc * 100:.2f}%")
                    print(f"Termination: prob={term_prob:.2f}"
                          f" ({'yes' if term_prob > 0.5 else 'no'}), "
                          f"target={'yes' if term_true.item() else 'no'}")
                    print()

                total_acc += acc
                total_steps += 1
                term_score += (term_true == (term_prob > 0.5))

                x = y_pred.detach()

                if term_prob > 0.5:
                    break

            final_comp_pred = [round(v) for v in y_pred.squeeze(1).tolist()]
            final_comp_true = [round(v) for v in states[-1]]
            total_last_step_acc += sum(p == t for p, t in zip(final_comp_pred, final_comp_true)) / len(final_comp_true)

    print(f"Total mean accuracy: {total_acc / total_steps * 100:.2f}%")
    print(f"Last step accuracy: {total_last_step_acc / len(data) * 100:.2f}%")
    print(f"Termination score: {term_score.item() / total_steps * 100:.2f}%")

def test(test_size: int = 1, num_nodes: int = 20, details: bool = True) -> None:
    algos = ['BFS', 'BF', 'PRIM', 'CC']
    print(algos)
    model = Model(algos, 1, 32, 1)
    #model = torch.compile(model)
    model.load_state_dict(torch.load("parameters/model.pt", weights_only=True))
    model.eval()

    test_data = []

    for _ in range(test_size):
        graph = random_graph(
            num_nodes=num_nodes,
            p=min(0.5, math.log2(num_nodes) / num_nodes),
            seed=None,
            self_loop=True,
            weighted=True,
            weight_mn=0.2,
            weight_mx=2.0
        )
        source = random.randrange(graph.num_nodes)
        test_data.append((graph, source))

    test_data = generate_training_graphs(test_size, num_nodes, weighted=True,
                                         weight_mn=0.2, weight_mx=2.0)

    device = next(model.parameters()).device

    details=True

    test_bfs(model, test_data, device, details)
    test_bf(model, test_data, device, details)
    test_prim(model, test_data, device, details)
    test_cc(model, test_data, device, details)

if __name__ == "__main__":
    test_size = 5
    num_nodes = 20
    test(test_size=test_size, num_nodes=num_nodes)