import os
import torch
from torch_geometric.loader import NeighborLoader
import copy

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


def make_reverse_neighbor_loader(data, num_neighbors=[15, 10, 5], batch_size=2048, shuffle=False, input_nodes=None):
    """
    Neighbor sampling expands neighborhoods using the given edge_index
    by duplicating the data and flipping its edge_index so the sampling is in backward direction.
    """
    data_rev = copy.copy(data)  
    data_rev.edge_index = data.edge_index[[1, 0], :]  # Reversed edge indices for sampling only
    loader = NeighborLoader(
        data_rev,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        shuffle=shuffle,
        input_nodes=input_nodes  
    )
    # Keep originals for later reconstruction of forward edges per batch:
    loader._full_edge_index_fwd = data.edge_index
    loader._num_nodes_full = data.num_nodes
    return loader


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
