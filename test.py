import random


from model import Model
from graph_generation import random_graph

import algorithms as algorithms

import torch
import torch.nn as nn
import torch.optim as optim

BCE = nn.BCEWithLogitsLoss()
MSE = nn.MSELoss()
CE = nn.CrossEntropyLoss()

def test_bfs(model, data:list, source:int, device, details=False) -> None:
    print("** TESTING BFS **")

    total_score = 0
    total_last_step_score = 0
    total_steps = 0

    for graph, source in data:

        states = algorithms.compute_bfs_states(source, graph)

        h = None

        print('State 0 = ', states[0])

        x = torch.tensor(states[0], dtype=torch.float32).unsqueeze(1).to(device)

        with torch.no_grad():
            for step in range(len(states) - 1):
                state = states[step]
                target = states[step + 1]
                y_true = torch.tensor(target, dtype=torch.float32).unsqueeze(1).to(device)
                term_true = torch.tensor([1 if step == (len(states) - 2) else 0])

                y_pred, p_pred, h, term_pred = model('BFS', graph, x, h)

                bce = BCE(y_pred, y_true).item()

                probs = torch.sigmoid(y_pred).squeeze(1).tolist()
                prediction = [1 if p > 0.5 else 0 for p in probs]

                visited_pred = [i for i, p in enumerate(probs) if p > 0.5]
                visited_true = [i for i, v in enumerate(target) if v == 1.0]
            
                score = sum(p == t for p, t in zip(prediction, target)) / len(target)

                h = h.detach()

                if details:
                    print(f"Step {step}")
                    for i in range(len(state)):
                        print(f"  node {i}: input={state[i]:.0f}  prob={probs[i]:.2f} "
                              f"prediction={prediction[i]} target={target[i]:.0f}")
                    print(f"Visited prediction: {visited_pred}")
                    print(f"Visited target:     {visited_true}")
                    print(f"Termination: prob={torch.sigmoid(term_pred).item():.2f}"
                          f"({'yes' if torch.sigmoid(term_pred).item() > 0.5 else 'no'}), " 
                          f"target={'yes' if step == len(states) - 2 else 'no'}")
                    print("BCE:", bce)
                    print(f"{score * 100:.2f}% predicted")
                    print()

                total_score += score
                total_steps += 1

                x = torch.sigmoid(y_pred).detach()

                if torch.sigmoid(term_pred).item() > 0.5: # terminate signal
                    break

            target = states[-1]
            score = sum(p == t for p, t in zip(prediction, target)) / len(target)
            total_last_step_score += score

    print(f"Total score: {total_score / total_steps * 100:.2f}%")
    print(f"Last step score: {total_last_step_score / len(data) * 100:.2f}%")

def test_bf(model, data: list, device, details=False) -> None:
    print("** TESTING BELLMAN-FORD **")

    total_mse = 0.0
    total_p_loss = 0.0

    total_last_step_mse = 0.0
    total_steps = 0

    for graph, source in data:
        states, preds, inf = algorithms.compute_bf_states(source, graph)

        if len(states) < 2:
            continue

        h = None
        x = torch.tensor(states[0], dtype=torch.float32).unsqueeze(1).to(device)

        with torch.no_grad():
            for step in range(len(states) - 1):
                state = states[step]
                target = states[step + 1]
                pred = preds[step + 1]

                y_true = torch.tensor(target, dtype=torch.float32).unsqueeze(1).to(device)
                p_true = torch.tensor(pred, dtype=torch.long).to(device)

                y_pred, p_pred, h, term_pred = model('BF', graph, x, h)

                p_pred_scores = torch.stack(p_pred)

                mse = MSE(y_pred, y_true).item()
                ce = CE(p_pred_scores, p_true).item()
                p_pred = p_pred_scores.argmax(dim=1).tolist()
                pred_acc = sum(p == t for p, t in zip(p_pred, pred)) / len(pred)

                h = h.detach()

                if details:
                    print(f"Step {step}")
                    for i in range(len(state)):
                        print(f"  node {i}: dist={state[i]:.2f} -> pred={y_pred.squeeze(1)[i].item():.2f} "
                              f"(target={target[i]:.2f})  "
                              f"predecessor: pred={p_pred[i]} target={pred[i]}")
                    print(f"MSE: {mse:.4f}  Predecessor CE: {ce:.4f}  Pred acc: {pred_acc*100:.1f}%")
                    print(f"Termination: prob={torch.sigmoid(term_pred).item():.2f}"
                          f"({'yes' if torch.sigmoid(term_pred).item() > 0.5 else 'no'}), "
                          f"target={'yes' if step == len(states) - 2 else 'no'}")
                    print()

                total_mse += mse
                total_p_loss += pred_acc
                total_steps += 1

                x = y_pred.detach()

                if torch.sigmoid(term_pred).item() > 0.5:
                    break

            final_target = states[-1]
            total_last_step_mse += MSE(y_pred, torch.tensor(final_target, dtype=torch.float32).unsqueeze(1).to(device)).item()

    print(f"Total mean MSE: {total_mse / total_steps:.4f}")
    print(f"Total mean predecessor acc: {total_p_loss / total_steps * 100:.2f}%")
    print(f"Last step mean MSE: {total_last_step_mse / len(data):.4f}")

def test_prim(model, data: list, device, details=False) -> None:
    print("** TESTING PRIM **")

    total_bce = 0.0
    total_pred_acc = 0.0
    total_last_step_bce = 0.0
    total_steps = 0

    for graph, source in data:
        states, preds = algorithms.compute_prim_states(source, graph)

        if len(states) < 2:
            continue

        h = None
        x = torch.tensor(states[0], dtype=torch.float32).unsqueeze(1).to(device)

        with torch.no_grad():
            for step in range(len(states) - 1):
                state = states[step]
                target = states[step + 1]
                pred = preds[step + 1]

                y_true = torch.tensor(target, dtype=torch.float32).unsqueeze(1).to(device)
                p_true = torch.tensor(pred, dtype=torch.long).to(device)

                y_pred, p_pred, h, term_pred = model('PRIM', graph, x, h)

                p_pred_scores = torch.stack(p_pred)

                bce = BCE(y_pred, y_true).item()
                ce = CE(p_pred_scores, p_true).item()
                p_pred = p_pred_scores.argmax(dim=1).tolist()
                pred_acc = sum(p == t for p, t in zip(p_pred, pred)) / len(pred)
                mst_pred = [i for i, v in enumerate(torch.sigmoid(y_pred).squeeze(1).tolist()) if v > 0.5]
                mst_true = [i for i, v in enumerate(target) if v == 1]

                h = h.detach()

                if details:
                    print(f"Step {step}")
                    print(f"  MST prediction: {mst_pred}  target: {mst_true}")
                    for i in range(len(state)):
                        print(f"  node {i}: in_mst={state[i]:.0f}  predecessor: pred={p_pred[i]} target={pred[i]}")
                    print(f"BCE: {bce:.4f}  Predecessor CE: {ce:.4f}  Pred acc: {pred_acc*100:.1f}%")
                    print(f"Termination: prob={torch.sigmoid(term_pred).item():.2f}"
                          f"({'yes' if torch.sigmoid(term_pred).item() > 0.5 else 'no'}), "
                          f"target={'yes' if step == len(states) - 2 else 'no'}")
                    print()

                total_bce += bce
                total_pred_acc += pred_acc
                total_steps += 1

                x = torch.sigmoid(y_pred).detach()

                if torch.sigmoid(term_pred).item() > 0.5:
                    break

            final_target = states[-1]
            total_last_step_bce += BCE(y_pred, torch.tensor(final_target, dtype=torch.float32).unsqueeze(1).to(device)).item()

    print(f"Total mean BCE: {total_bce / total_steps:.4f}")
    print(f"Total mean predecessor acc: {total_pred_acc / total_steps * 100:.2f}%")
    print(f"Last step mean BCE: {total_last_step_bce / len(data):.4f}")

def test_dfs(model, data: list, device, details=False) -> None:
    print("** TESTING DFS **")

    total_score = 0
    total_last_step_score = 0
    total_steps = 0

    for graph, source in data:
        states = algorithms.compute_dfs_states(source, graph)

        h = None
        x = torch.tensor(states[0], dtype=torch.float32).unsqueeze(1).to(device)

        with torch.no_grad():
            for step in range(len(states) - 1):
                state = states[step]
                target = states[step + 1]

                y_true = torch.tensor(target, dtype=torch.float32).unsqueeze(1).to(device)

                y_pred, h, term_pred = model('DFS', graph, x, h)

                bce = BCE(y_pred, y_true).item()
                probs = torch.sigmoid(y_pred).squeeze(1).tolist()
                prediction = [1 if p > 0.5 else 0 for p in probs]
                visited_pred = [i for i, p in enumerate(probs) if p > 0.5]
                visited_true = [i for i, v in enumerate(target) if v == 1.0]
                score = sum(p == t for p, t in zip(prediction, target)) / len(target)

                h = h.detach()

                if details:
                    print(f"Step {step}")
                    for i in range(len(state)):
                        print(f"  node {i}: input={state[i]:.0f}  prob={probs[i]:.2f} "
                              f"prediction={prediction[i]} target={target[i]:.0f}")
                    print(f"Visited prediction: {visited_pred}")
                    print(f"Visited target:     {visited_true}")
                    print(f"Termination: prob={torch.sigmoid(term_pred).item():.2f}"
                          f"({'yes' if torch.sigmoid(term_pred).item() > 0.5 else 'no'}), "
                          f"target={'yes' if step == len(states) - 2 else 'no'}")
                    print("BCE:", bce)
                    print()

                total_score += score
                total_steps += 1

                x = torch.sigmoid(y_pred).detach()

                if torch.sigmoid(term_pred).item() > 0.5:
                    break

            final_target = states[-1]
            total_last_step_score += sum(p == t for p, t in zip(prediction, final_target)) / len(final_target)

    print(f"Total score: {total_score / total_steps * 100:.2f}%")
    print(f"Last step score: {total_last_step_score / len(data) * 100:.2f}%")


def test_dijkstra(model, data: list, device, details=False) -> None:
    print("** TESTING DIJKSTRA **")

    total_mse = 0.0
    total_pred_acc = 0.0
    total_last_step_mse = 0.0
    total_steps = 0

    for graph, source in data:
        states, preds, _ = algorithms.compute_dijkstra_states(source, graph)

        if len(states) < 2:
            continue

        h = None
        x = torch.tensor(states[0], dtype=torch.float32).unsqueeze(1).to(device)

        with torch.no_grad():
            for step in range(len(states) - 1):
                state = states[step]
                target = states[step + 1]
                pred = preds[step + 1]

                y_true = torch.tensor(target, dtype=torch.float32).unsqueeze(1).to(device)
                p_true = torch.tensor(pred, dtype=torch.long).to(device)

                y, h, term_pred = model('Dijkstra', graph, x, h)

                y_pred = y[0]
                p_pred_scores = torch.stack(y[1])

                mse = MSE(y_pred, y_true).item()
                ce = CE(p_pred_scores, p_true).item()
                p_pred = p_pred_scores.argmax(dim=1).tolist()
                pred_acc = sum(p == t for p, t in zip(p_pred, pred)) / len(pred)

                h = h.detach()

                if details:
                    print(f"Step {step}")
                    for i in range(len(state)):
                        print(f"  node {i}: dist={state[i]:.2f} -> pred={y_pred.squeeze(1)[i].item():.2f} "
                              f"(target={target[i]:.2f})  "
                              f"predecessor: pred={p_pred[i]} target={pred[i]}")
                    print(f"MSE: {mse:.4f}  Predecessor CE: {ce:.4f}  Pred acc: {pred_acc*100:.1f}%")
                    print(f"Termination: prob={torch.sigmoid(term_pred).item():.2f}"
                          f"({'yes' if torch.sigmoid(term_pred).item() > 0.5 else 'no'}), "
                          f"target={'yes' if step == len(states) - 2 else 'no'}")
                    print()

                total_mse += mse
                total_pred_acc += pred_acc
                total_steps += 1

                x = y_pred.detach()

                if torch.sigmoid(term_pred).item() > 0.5:
                    break

            final_target = states[-1]
            total_last_step_mse += MSE(y_pred, torch.tensor(final_target, dtype=torch.float32).unsqueeze(1).to(device)).item()

    print(f"Total mean MSE: {total_mse / total_steps:.4f}")
    print(f"Total mean predecessor acc: {total_pred_acc / total_steps * 100:.2f}%")
    print(f"Last step mean MSE: {total_last_step_mse / len(data):.4f}")


def test(test_size:int = 1, num_nodes:int = 20):
    algos = ['BFS', 'BF', 'PRIM']
    model = Model(algos, 1, 32, 1)
    model.load_state_dict(torch.load("parameters/model.pt", weights_only=True))
    model.eval()

    test_data = []

    for _ in range(test_size):
        graph = random_graph(
            num_nodes=num_nodes,
            p=0.3,
            seed=None,
            self_loop=True,
            weighted=True,
            weight_mn=0.2,
            weight_mx=2.0
        )
        source = random.randrange(graph.num_nodes)
        test_data.append((graph, source))

    device = device = next(model.parameters()).device
    details=True

    test_bfs(model, test_data, source, device, details)
    test_bf(model, test_data, device, details)
    test_prim(model, test_data, device, details)
    #test_dfs(model, graph, source)
    #test_dijkstra(model, graph, source)

if __name__ == "__main__":
    test_size = 5
    num_nodes = 10
    test(test_size=test_size, num_nodes=num_nodes)