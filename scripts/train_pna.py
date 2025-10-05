#!/usr/bin/env python3
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.utils import degree

from utils.metrics import append_f1_score_to_csv
from utils.seed import set_seed
from models.pna_baseline import PNANet

BEST_MODEL_PATH = "./checkpoints/pna_baseline"
DATA_PATH = "./data"
 
def load_datasets(log_dir=DATA_PATH, train_data_file="train.pt", val_data_file="val.pt", test_data_file="test.pt"):
    train = torch.load(os.path.join(log_dir, train_data_file), weights_only=False, map_location="cpu")
    val= torch.load(os.path.join(log_dir, val_data_file), weights_only=False, map_location="cpu")
    test = torch.load(os.path.join(log_dir, test_data_file), weights_only=False, map_location="cpu")
    return train, val, test


def ensure_node_features(g):
    '''
    Ensure that the graph has node features.
    The Provably Powerful GNNs paper uses constant node features in its baseline models.
    As a baseline approach, assign ones to each node.
    '''
    if getattr(g, 'x', None) is None:
        N = g.y.shape[0] if getattr(g, 'y', None) is not None else int(g.edge_index.max()) + 1
        # Assign constant features (all ones) to each node
        g.x = torch.ones((N, 1), dtype=torch.float)
    return g


def compute_minority_f1_score_per_task(logits, labels, threshold=0.5):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold)
    y = labels.bool()

    N, C = y.shape
    f1_scores = torch.zeros(C, dtype=torch.float32, device=logits.device)
    epsilon = 1e-12
    
    for c in range(C):
        y_c = y[:, c]

        # Find the minority class (either 0 or 1)
        pos = y_c.sum()
        neg = y_c.numel() - pos
        minority_is_one = (pos <= neg) 

        if minority_is_one:
            y_pos    = y_c
            pred_pos = preds[:, c]
        else:
            y_pos    = ~y_c
            pred_pos = ~preds[:, c]

        true_pos = (y_pos & pred_pos).sum().float()
        false_pos = ((~y_pos) & pred_pos).sum().float()
        false_neg = (y_pos & (~pred_pos)).sum().float()
        
        precision = true_pos / (true_pos + false_pos + epsilon)
        recall = true_pos / (true_pos + false_neg + epsilon)
        f1 = 2 * precision * recall / (precision + recall + epsilon)
        f1_scores[c] = f1

    return f1_scores


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_nodes = 0

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()

        out = model(data.x, data.edge_index) # Forward pass
        loss = criterion(out, data.y.float())  # Binary cross-entropy loss for multi-label
        loss.backward()
        optimizer.step()
        
        total_loss  += loss.item() * int(data.num_nodes)
        total_nodes += int(data.num_nodes)

    # Take the mean loss per node, per task
    average_loss = total_loss / max(total_nodes, 1)

    return average_loss


@torch.no_grad()
def evaluate_epoch(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_nodes = 0
    total_pairs = 0
    correct_pairs = 0

    all_logits = []
    all_labels = []

    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index)            
        loss = criterion(out, data.y.float())    

        total_loss += loss.item() * int(data.num_nodes)
        total_nodes += int(data.num_nodes)

        preds = (torch.sigmoid(out) > 0.5) # Turn logits into binary predictions
        correct_pairs += (preds == data.y.bool()).sum().item()
        total_pairs += data.y.numel()

        all_logits.append(out)
        all_labels.append(data.y)

    avg_loss = total_loss / max(total_nodes, 1)
    per_node_acc = correct_pairs / max(total_pairs, 1)  

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    f1_score_per_task = compute_minority_f1_score_per_task(logits, labels)

    return avg_loss, per_node_acc, f1_score_per_task


def run_pna(seed, device):
    set_seed(seed)

    train_data, val_data, test_data = load_datasets()

    # Assign constant features
    train_data = ensure_node_features(train_data)
    val_data = ensure_node_features(val_data)
    test_data = ensure_node_features(test_data)

    d = degree(train_data.edge_index[1], num_nodes=train_data.num_nodes).long()
    deg_hist = torch.bincount(d, minlength=int(d.max()) + 1)

    # Define the model
    in_dim = train_data.num_node_features if train_data.x is not None else 1
    out_dim = train_data.y.size(-1)
    model = PNANet(in_dim, hidden_dim=64, out_dim=out_dim, deg=deg_hist, num_layers=6, dropout=0.1).to(device)

    # Load the datasets
    # Note: Because we currently have only 1 graph per split, there is no need for batching.
    train_loader = [train_data]
    valid_loader = [val_data]
    test_loader  = [test_data]

    # Define optimizer and loss functions
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4) # Define optimizer as Adam
    criterion = nn.BCEWithLogitsLoss() # Define loss as binary cross-entropy (preferred for multi-label classification task we have here)

    # Training loop
    best_val = float("inf")
    for epoch in range(1, 101):  # a few more epochs helps stabilize F1
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, _, val_f1 = evaluate_epoch(model, valid_loader, criterion, device)
        val_macro = val_f1.mean().item()

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), os.path.join(BEST_MODEL_PATH, f"best_pna_seed{seed}.pt"))

        if epoch % 10 == 0:
            print(f"[seed {seed}] Epoch {epoch:03d} | train {train_loss:.4f} | val {val_loss:.4f} | val macro-minF1 {100*val_macro:.2f}%")

    # Save the best model and evaluate on test dataset
    model.load_state_dict(torch.load(os.path.join(BEST_MODEL_PATH, f"best_pna_seed{seed}.pt"), map_location=device))
    test_loss, _, test_f1 = evaluate_epoch(model, test_loader, criterion, device)
    return test_loss, test_f1  


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the sub-tasks
    tasks = ["deg-in","deg-out","fan-in","fan-out","C2","C3","C4","C5","C6","S-G","B-C"]

    seeds = [0,1,2,3,4]
    test_f1_scores = []
    for s in seeds:
        _, test_f1 = run_pna(s, device)
        test_f1_scores.append(test_f1.cpu())

    all_f1 = torch.stack(test_f1_scores, dim=0)        
    mean_f1 = all_f1.mean(dim=0)              
    std_f1  = all_f1.std(dim=0, unbiased=False)

    macro_mean = mean_f1.mean().item()*100
    print(f"\nPNA baseline — macro minority F1 over 5 runs: {macro_mean:.2f}%")

    row = " | ".join(f"{n}: {100*m:.2f}±{100*s:.2f}%" for n, m, s in zip(tasks, mean_f1.tolist(), std_f1.tolist()))
    print("Per-task (mean±std over 5 runs):", row)

    # Append F1 scores to CSV
    append_f1_score_to_csv(
        out_csv="./results/metrics/f1_scores.csv",
        tasks=tasks,
        mean_f1=mean_f1,
        std_f1=std_f1,
        macro_mean_percent=macro_mean,
        seeds=seeds,
        model_name="PNA baseline",
    )


if __name__ == "__main__":
    main()
