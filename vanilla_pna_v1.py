#!/usr/bin/env python3
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import PNAConv, BatchNorm, global_mean_pool
from torch_geometric.loader import DataLoader
from torch_geometric.utils import degree

from utils.gcn_utils import GraphData  


'''
The PNANet class described in this script is defined based on the following paper:
https://arxiv.org/pdf/2004.05718
'''

BEST_MODEL_PATH = "./logs/model_seeds"
BEST_MODEL_NAME = "best_pna_v1.pt"

def load_datasets(log_dir="./data", train_data_file="train.pt", val_data_file="val.pt", test_data_file="test.pt"):
    train = torch.load(os.path.join(log_dir, train_data_file), weights_only=False, map_location="cpu")
    val= torch.load(os.path.join(log_dir, val_data_file), weights_only=False, map_location="cpu")
    test = torch.load(os.path.join(log_dir, test_data_file), weights_only=False, map_location="cpu")
    return train, val, test


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


class PNANet(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, deg):
        super().__init__()

        aggregators = ['mean', 'min', 'max', 'std']
        scalers = ['amplification', 'attenuation', 'identity']

        # 1st layer does 1-hop message passing
        self.conv1 = PNAConv(in_dim, hidden_dim, aggregators, scalers, deg=deg, 
                             towers=4, pre_layers=1, post_layers=1, divide_input=False) 
        self.bn1 = BatchNorm(hidden_dim)

        # 2nd layer does 2-hops message passing
        self.conv2 = PNAConv(hidden_dim, hidden_dim, aggregators, scalers, deg=deg, 
                             towers=4, pre_layers=1, post_layers=1, divide_input=False) 
        self.bn2 = BatchNorm(hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim) # define per-node logits (because my labels are per-node - so I want to predict per-node labels)
        )

    def forward(self, x, edge_index, batch=None):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)

        return self.mlp(x) # [num_nodes, out_dim]


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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the sub-tasks
    tasks = ["deg-in","deg-out","fan-in","fan-out","C2","C3","C4","C5","C6","S-G","B-C"]

    train_data, val_data, test_data = load_datasets()

    d = degree(train_data.edge_index[1], num_nodes=train_data.num_nodes).long()
    deg = torch.bincount(d, minlength=int(d.max()) + 1)

    # Define the model
    in_dim = train_data.num_node_features if train_data.x is not None else 1
    out_dim = train_data.y.size(-1)
    model = PNANet(in_dim, hidden_dim=64, out_dim=out_dim, deg=deg).to(device)

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
    for epoch in range(1, 51):
        loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_f1_per_task = evaluate_epoch(model, valid_loader, criterion, device)
        val_macro_f1 = val_f1_per_task.mean().item() * 100.0
        print(f"Epoch {epoch:03d} | Train Loss {loss:.4f} | Val Loss {val_loss:.4f} | Val Acc {val_acc:.4f} | Val Minority F1 (macro) {val_macro_f1:.2f}%")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), os.path.join(BEST_MODEL_PATH, BEST_MODEL_NAME))

    # Evaluate on the test dataset
    model.load_state_dict(torch.load(os.path.join(BEST_MODEL_PATH, BEST_MODEL_NAME)))
    test_loss, test_acc, test_f1_per_task = evaluate_epoch(model, test_loader, criterion, device)
    test_macro_f1 = test_f1_per_task.mean().item() * 100.0
    
    print(f"Test Loss {test_loss:.4f} | Test Accuracy {test_acc:.4f} | Test Minority F1 (macro) {test_macro_f1:.2f}%")
    print("Per-task (test) →", " | ".join(f"{n}: {100*v:.2f}%" for n, v in zip(tasks, test_f1_per_task.cpu().tolist())))


if __name__ == "__main__":
    main()
