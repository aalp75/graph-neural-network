import math
import random

import torch
import torch.nn as nn

from model import Model
from graph_generation import random_graph, generate_training_graphs
import algorithms

def generate_test_data(test_size: int, num_nodes: int) -> tuple:

    data = []
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
        data.append((graph, source))
    
    return data

def test_bfs(model: Model, data: list, device: torch.device, details: bool = False) -> None:

    total_score = 0.0
    term_score = 0.0
    last_step_score = 0.0

    total_steps = 0

    for ite, (graph, source) in enumerate(data):
        if details: print(f"-- Iteration {ite} --")
        states, predecessors, _, termination = algorithms.compute_states('bfs', graph, source)
        steps = algorithms.generate_steps(states, predecessors, termination)

        x = torch.tensor(states[0], dtype=torch.float32).unsqueeze(1).to(device)
        h = torch.zeros(graph.num_nodes, model.hidden_dim, device=device)

        edges = graph.get_edge_tensors(device)

        for step, (state, next_state, predecessors, termination) in enumerate(steps):

            if details:
                print(f"Initial state: {states[0][0]}\n")    

            y_pred, _, h, term_pred = model('bfs', edges, x, h)

            y_prob = torch.sigmoid(y_pred)
            prediction = [1 if p > 0.5 else 0 for p in y_prob.squeeze(1)]
            score = sum([s1 == s2 for s1, s2 in zip(prediction, next_state)]) / len(next_state)
            total_score += score

            term_prob = torch.sigmoid(term_pred).item()
            term_pred_str = "yes" if term_prob > 0.5 else "no"
            term_str = "yes" if termination == 1 else "no"
            term_score += ((term_prob > 0.5) == termination)

            total_steps += 1

            if details:
                print(f"Step {step}")
                print(f"Prediction: {prediction}")
                print(f"Target:     {next_state}")
                print(f"Termination: prob={term_prob:.2f}"
                      f" ({term_pred_str}), target={term_str}")
                print(f"{score * 100:.2f}% predicted")
                print()

            #if term_prob > 0.5:
                #break

            h = h.detach()
            x = torch.sigmoid(y_pred).detach()

        target = states[-1]
        score = sum(p == t for p, t in zip(prediction, target)) / len(target)
        last_step_score += score

    print(f"Score: {total_score / total_steps * 100:.2f}% (Accuracy)")
    print(f"Last step score: {last_step_score / len(data) * 100:.2f}% (Accuracy)")
    print(f"Termination score: {term_score / total_steps * 100:.2f}% (Accuracy)")

def test_bf(model: Model, data: list, device: torch.device, details: bool = False) -> None:

    total_score = 0
    term_score = 0
    predec_score = 0

    total_steps = 0

    for graph, source in data:
        edges = graph.get_edge_tensors(device)
        states, predecessors, inf, terminations = algorithms.compute_states('bf', graph, source)
        steps = algorithms.generate_steps(states, predecessors, terminations)

        h = torch.zeros(graph.num_nodes, model.hidden_dim)
        x = torch.tensor(states[0], dtype=torch.float32).unsqueeze(1).to(device)

        for step, (state, next_state, predecessors, termination) in enumerate(steps):
            y_true = torch.tensor(next_state, dtype=torch.float32).unsqueeze(1)
            y_predict, p_predict, h, term_predict = model('bf', edges, x, h)

            score = torch.nn.MSELoss()(y_predict, y_true)
            total_score += score

            term_proba = torch.sigmoid(term_predict).item()
            term_predict = 1 if term_proba > 0.5 else 0
            term_predict_str = "yes" if term_proba > 0.5 else "no"
            term_str = "yes" if termination == 1 else "no"

            term_score += (term_predict == termination)

            reachable = [v < inf and node != source for node, v in enumerate(next_state)]
            reachable_t = torch.tensor(reachable, dtype=torch.bool)
            predecessors_t = torch.tensor(predecessors, dtype=torch.long)

            predecessor_predict = torch.full((graph.num_nodes,), -1, dtype=torch.long)
            if any(reachable):
                predecessor_predict[reachable_t] = p_predict[reachable_t].argmax(dim=1)
                predec_score += (predecessor_predict[reachable_t] == predecessors_t[reachable_t]).float().mean().item()

            total_steps += 1

            if details:
                print(f"Step {step + 1}")
                for node in range(graph.num_nodes):
                    print(f"Prediction node {node} dist = {y_predict.squeeze(1)[node]:.2f} ({next_state[node]:.2f}) "
                        f"predecessor = {predecessor_predict[node]} ({predecessors[node]})")
                print(f"Score = {score:.4f} (MSE)")
                print(f"Termination probability = {term_proba:.2f} ({term_predict_str}) exepected: {term_str}")
                print()

            #x = torch.tensor(next_state).unsqueeze(1)
            x = y_predict.detach()
            h = h.detach()

    print(f"Score = {total_score / total_steps:.4f} (MSE)")
    print(f"Termination score = {term_score / total_steps * 100:.2f} (Accuracy %)")
    print(f"Predecessor score = {predec_score / total_steps * 100:.2f} (Accuracy %)")


def test_prim(model: Model, data: list, device: torch.device, details: bool = False) -> None:

    score = 0.0 # next node prediction

    total_ce = 0.0
    total_next_node_acc = 0.0
    total_p_acc = 0.0
    total_last_step_p_acc = 0.0
    total_steps = 0
    term_score = 0

    for graph, source in data:
        states, parents = algorithms.compute_prim_states(graph, source)

        if details:
                print(f"Initial state: {states[0][0]}\n") 

        h = torch.zeros(graph.num_nodes, model.hidden_dim, device=device)
        edges = graph.get_edge_tensors(device)
        x = torch.tensor(states[0], dtype=torch.float32).unsqueeze(1).to(device)

        with torch.no_grad():
            for step in range(len(states) - 1):
                state = states[step]
                #x = torch.tensor(state, dtype=torch.float32).unsqueeze(1).to(device)
                target = states[step + 1]
                parent = parents[step + 1]
                next_node = next(i for i in range(graph.num_nodes) if state[i] == 0 and target[i] == 1)

                next_node_true = torch.tensor([next_node], dtype=torch.long).to(device)
                p_true = torch.tensor(parent, dtype=torch.long).to(device)
                term_true = torch.tensor([1 if step == (len(states) - 2) else 0])

                y_pred, p_pred, h, term_pred = model('prim', edges, x, h)
                term_prob = torch.sigmoid(term_pred).item()

                reachable = [i for i in range(graph.num_nodes) if target[i] == 1 and i != source]
                ce = nn.CrossEntropyLoss()(y_pred.squeeze(1).unsqueeze(0), next_node_true).item()
                p_ce = nn.CrossEntropyLoss()(p_pred[reachable], p_true[reachable]).item() if reachable else 0.0
                logits = y_pred.squeeze(1)
                undiscovered = [i for i in range(graph.num_nodes) if state[i] == 0]
                next_node_pred = max(undiscovered, key=lambda i: logits[i].item())
                p_pred_list = p_pred.argmax(dim=1).tolist()  # best predecessor of each node i

                p_acc = sum(p_pred_list[i] == parent[i] for i in reachable) / len(reachable) if reachable else 1.0

                h = h.detach()

                if details:
                    mst_pred = sorted([i for i, v in enumerate(state) if v == 1] + [next_node_pred])
                    mst_true = [i for i, v in enumerate(target) if v == 1]
                    print(f"Step {step}")
                    print(f"MST prediction: {mst_pred}")
                    print(f"MST target:     {mst_true}")
                    print(f"Next node: predicted: {next_node_pred}  (target: {next_node})")
                    print(f"Next node CE: {ce:.4f}  Predecessor CE: {p_ce:.4f}  Pred acc: {p_acc*100:.1f}%")
                    print(f"Termination: prob={term_prob:.2f}"
                          f" ({'yes' if term_prob > 0.5 else 'no'}), "
                          f"target={'yes' if term_true.item() else 'no'}")
                    print()

                total_ce += ce
                score += (next_node_pred == next_node)
                total_p_acc += p_acc
                total_steps += 1
                term_score += (term_true == (term_prob > 0.5))

                x = x.clone()
                x[next_node_pred] = 1.0

                #if term_prob > 0.5:
                    #break

            final_parent = parents[-1]
            final_reachable = [i for i in range(graph.num_nodes) if states[-1][i] == 1 and i != source]
            final_p_pred = p_pred.argmax(dim=1).tolist()
            total_last_step_p_acc += sum(final_p_pred[i] == final_parent[i] for i in final_reachable) / len(final_reachable) if final_reachable else 1.0

    print(f"Score: {score / total_steps * 100:.2f}% (Next node prediction accucary)")
    print(f"Total mean predecessor acc: {total_p_acc / total_steps * 100:.2f}%")
    print(f"Last step predecessor acc: {total_last_step_p_acc / len(data) * 100:.2f}%")
    print(f"Termination score: {term_score.item() / total_steps * 100:.2f}%")

def test_cc(model: Model, data: list, device: torch.device, details: bool = False) -> None:
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
        edges = graph.get_edge_tensors(device)
        x = torch.tensor(states[0], dtype=torch.float32).unsqueeze(1).to(device)

        with torch.no_grad():
            for step in range(len(states) - 1):
                state = states[step]
                target = states[step + 1]

                term_true = torch.tensor([1 if step == (len(states) - 2) else 0])

                y_pred, _, h, term_pred = model('cc', edges, x, h)
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

    print(f"Score: {total_acc / total_steps * 100:.2f}% (Accuracy)")
    print(f"Last step accuracy: {total_last_step_acc / len(data) * 100:.2f}%")
    print(f"Termination score: {term_score.item() / total_steps * 100:.2f}% (Accuracy)")

if __name__ == "__main__":
    algos = ['bfs', 'bf', 'prim', 'cc']
    #algos = ['bf']
    model = Model(algos, 1, 32, 1)
    #model = torch.compile(model)
    #model.load_state_dict(torch.load("parameters/model.pt", weights_only=True))
    model.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in torch.load("parameters/modelgood.pt", weights_only=True).items()})
    model.eval()

    test_size = 5
    num_nodes = 100
    details = False

    test_data = generate_test_data(test_size, num_nodes)

    device = next(model.parameters()).device

    #test_bfs(model, test_data, device, details)
    #test_bf(model, test_data, device, details)
    #test_prim(model, test_data, device, details)
    test_cc(model, test_data, device, details)