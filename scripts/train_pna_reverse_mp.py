#!/usr/bin/env python3
import os
import torch
import torch.nn as nn
from torch_geometric.utils import degree

from utils.metrics import append_f1_score_to_csv, start_epoch_csv, append_epoch_csv
from utils.seed import set_seed
from utils.train_utils import load_datasets, ensure_node_features, train_epoch, evaluate_epoch
from utils.hetero import make_bidirected_hetero
from models.pna_reverse_mp import PNANetReverseMP, compute_directional_degree_hists

BEST_MODEL_PATH = "./checkpoints/pna_reverse_mp"
MODEL_NAME = "pna_reverse_mp"
 
def run_pna(seed, tasks, device):
    set_seed(seed)

    train_data, val_data, test_data = load_datasets()

    # Assign constant features
    train_data = ensure_node_features(train_data)
    val_data = ensure_node_features(val_data)
    test_data = ensure_node_features(test_data)

    # Convert the data into HeteroData format
    # using forward and backward edge relations
    train_h = make_bidirected_hetero(train_data)
    val_h   = make_bidirected_hetero(val_data)
    test_h  = make_bidirected_hetero(test_data)

    # ---- Degree histograms per direction
    deg_fwd_hist, deg_rev_hist = compute_directional_degree_hists(
        edge_index=train_data.edge_index,  # original edges
        num_nodes=train_data.num_nodes,
    )

    # Define the model
    in_dim = train_h['n'].x.size(-1) if 'x' in train_h['n'] else 1
    out_dim = train_h['n'].y.size(-1)

    # Define the number of layers
    num_layers = 2
    print(f"Number of layers using in training: {num_layers}")
    
    model = PNANetReverseMP(
        in_dim=in_dim,
        hidden_dim=64,
        out_dim=out_dim,
        deg_fwd=deg_fwd_hist,
        deg_rev=deg_rev_hist,
        num_layers=num_layers,
        dropout=0.1,
        combine="sum",   # other aggregation options: 'mean' or 'max'
    ).to(device)

    # Load the hetero datasets
    train_loader = [train_h]
    valid_loader = [val_h]
    test_loader  = [test_h]

    # Define optimizer and loss functions
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4) # Define optimizer as Adam
    criterion = nn.BCEWithLogitsLoss() # Define loss as binary cross-entropy (preferred for multi-label classification task we have here)

    # Log the epoch results
    os.makedirs(BEST_MODEL_PATH, exist_ok=True)

    epoch_csv_path = start_epoch_csv(
        model_name=MODEL_NAME,
        seed=seed,
        tasks=tasks,
        out_dir=f"./results/metrics/epoch_logs/{MODEL_NAME}"
    )

    # Training loop
    best_val = float("inf")
    for epoch in range(1, 101):  # a few more epochs helps stabilize F1
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, _, val_f1 = evaluate_epoch(model, valid_loader, criterion, device)

        append_epoch_csv(epoch_csv_path, epoch, train_loss, val_loss, val_f1)

        val_macro = val_f1.mean().item()

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), os.path.join(BEST_MODEL_PATH, f"best_pna_reverse_mp_seed{seed}.pt"))

        if epoch % 10 == 0:
            print(f"[seed {seed}] Epoch {epoch:03d} | train {train_loss:.4f} | val {val_loss:.4f} | val macro-minF1 {100*val_macro:.2f}%")

    # Save the best model and evaluate on test dataset
    model.load_state_dict(torch.load(os.path.join(BEST_MODEL_PATH, f"best_pna_reverse_mp_seed{seed}.pt"), map_location=device))
    test_loss, _, test_f1 = evaluate_epoch(model, test_loader, criterion, device)
    return test_loss, test_f1  


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the sub-tasks
    tasks = ["deg-in","deg-out","fan-in","fan-out","C2","C3","C4","C5","C6","S-G","B-C"]

    seeds = [0,1,2,3,4]
    test_f1_scores = []
    for s in seeds:
        _, test_f1 = run_pna(s, tasks, device)
        test_f1_scores.append(test_f1.cpu())

    all_f1 = torch.stack(test_f1_scores, dim=0)        
    mean_f1 = all_f1.mean(dim=0)              
    std_f1  = all_f1.std(dim=0, unbiased=False)

    macro_mean = mean_f1.mean().item()*100
    print(f"\nPNA iterative reverse message passing — macro minority F1 over 5 runs: {macro_mean:.2f}%")

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
        model_name="PNA iterative reverse MP",
    )


if __name__ == "__main__":
    main()
