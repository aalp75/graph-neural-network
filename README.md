PyTorch implementation of the paper **Neural Execution of Graph Algorithms** ,Petar Veličković, Rex Ying, Matilde Padovano, Raia Hadsell, Charles Blundell (2019)

## Installation

```bash
git clone https://github.com/aalp75/neural-execution-graph-algorithms.git
cd neural-execution-graph-algorithms
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```python
>>> from model import Model
>>> from train import train

>>> model = Model(['bfs', 'bf', 'prim', 'cc'], in_dim=1, hidden_dim=32, out_dim=1)
>>> train(model, train_size=40, val_size=10, num_nodes=10, num_epochs=50, patience=10)

Epoch 1 | train loss = 0.1157 (state=0.0137, parent=0.0671, term=0.6977) | val loss = 0.0808 (state=0.0027, parent=0.0503, term=0.5571)
Epoch 2 | train loss = 0.0667 (state=0.0016, predec=0.0462, term=0.3772) | val loss = 0.0586 (state=0.0007, predec=0.0431, term=0.2964)
...

>>> prediction = model('bfs', edges, x, h)
```

More examples are available in the notebook ```example.ipynb```

## Results

### Bellman-Ford (MSE / Termination accuracy)

| Model MPNN-max | 20 nodes | 50 nodes | 100 nodes |
|---|---|---|---|
| Paper | 0.005 / 98.89% | 0.013 / 98.58% | 0.238 / 97.82% |
| Repository | 0.013 / 93.33% | 0.034 / 89.99% | 0.317 / 86.57% |

More details on the implementation and results can be found in this [report](https://github.com/aalp75/neural-execution-graph-algorithms/blob/main/report/report.pdf).

## References

Petar Veličković, Rex Ying, Matilde Padovano, Raia Hadsell, Charles Blundell (2019). Neural Execution of Graph Algorithms.

Petar Veličković, Adrià Puigdomènech Badia, David Budden, Razvan Pascanu, Andrea Banino, Misha Dashevskiy, Raia Hadsell, Charles Blundell (2022). The CLRS Algorithmic Reasoning Benchmark.

Justin Gilmer, Samuel S. Schoenholz, Patrick F. Riley, Oriol Vinyals, and George E. Dahl (2017),
Neural message passing for quantum chemistry.