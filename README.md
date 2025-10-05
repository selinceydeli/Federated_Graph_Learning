# Federated_Graph_Learning

## Synthetic Graph Generation

This repository includes a **synthetic subgraph-detection dataset** used for benchmarking graph models for the pattern detection task. The graphs and labels are generated following the pseudocode/configurations described in _Provably Powerful Graph Neural Networks for Directed Multigraphs_ (Egressy et al.).

### What’s included

- Three splits: **train**, **val**, **test**
- Saved as PyTorch tensors under `./data/`:

  - `train.pt`, `val.pt`, `test.pt` objects with node-level labels
  - `y_sums.csv` — per-split counts of positive labels per sub-task
  - `label_percentages.csv` — per-split label percentages and mean across splits, useful to sanity-check against the paper’s reported marginals

Each node is labeled for the presence of the following patterns (11 sub-tasks):

- `deg_in > 3`
- `deg_out > 3`
- `fan_in > 3`
- `fan_out > 3`
- `cycle2`
- `cycle3`
- `cycle4`
- `cycle5`
- `cycle6`
- `scatter_gather`
- `biclique`

### Reproducibility

Graph instances are **reproducible**. A single `BASE_SEED` deterministically derives distinct seeds for each split (train/val/test), ensuring:

- different graphs **within** a run for the splits,
- identical graphs **across** runs with the same `BASE_SEED`.

### Default generation settings

The default config (see the generator script `generate_synthetic.py`) follows the paper’s setup:

- Nodes `n = 8192`
- Average degree `d = 6`
- Radius parameter `r = 11.1`
- Directed multigraphs (for directed cycles)
- Generator: “chordal” / random-circulant-like
- One connected component per split (prevents data leakage)

### How to generate

From the repo root:

```bash
# 1) Generate graphs and labels
python3 generate_synthetic.py
```

After step (1), you’ll find `train.pt`, `val.pt`, `test.pt`, `y_sums.csv`, and `label_percentages.csv` under `./data/`.
