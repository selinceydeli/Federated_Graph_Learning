import os
import torch
from torch_geometric.loader import NeighborLoader
from torch_geometric.data import HeteroData
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


def _unpack_io(batch):
    """
    Helper method to unpack the batch into x_in, edge_in, y_true
    by differentiating between homogeneous and heterogeneous graphs

    Returns a tuple of (x_in, edge_in, y_true, num_nodes, is_hetero)
      - if homogeneous: edge_in is a Tensor edge_index
      - if hetero: edge_in is a dict {edge_type: edge_index}
    """
    is_hetero = isinstance(batch, HeteroData)
    if is_hetero:
        x_in = batch['n'].x
        y_true = batch['n'].y
        edge_in = {
            ('n','fwd','n'): batch[('n','fwd','n')].edge_index,
            ('n','rev','n'): batch[('n','rev','n')].edge_index,
        }
        num_nodes = int(batch['n'].num_nodes)
    else:
        x_in = batch.x
        y_true = batch.y
        edge_in = batch.edge_index
        num_nodes = int(batch.num_nodes)
    return x_in, edge_in, y_true, num_nodes, is_hetero


def train_epoch(model, loader, optimizer, criterion, device):
    """
    This method can be used for training both homogeneous and heterogeneous graphs
    """
    model.train()
    total_loss = 0.0
    total_nodes = 0

    for batch in loader:
        batch = batch.to(device)
        x_in, edge_in, y_true, n_nodes, is_hetero = _unpack_io(batch)

        optimizer.zero_grad()
        # Hetero model expect dict inputs; homogeneous expects tensors
        if is_hetero:
            out = model(x_in, edge_in)     # (x_dict['n'], edge_index_dict)
        else:
            out = model(x_in, edge_in)     # (x, edge_index)

        loss = criterion(out, y_true.float())
        loss.backward()
        optimizer.step()

        total_loss  += loss.item() * n_nodes
        total_nodes += n_nodes

    return total_loss / max(total_nodes, 1)


@torch.no_grad()
def evaluate_epoch(model, loader, criterion, device):
    """
    This method can be used for evaluating both homogeneous and heterogeneous graphs
    """
    model.eval()

    total_loss = 0.0
    total_nodes = 0
    total_pairs = 0
    correct_pairs = 0

    all_logits = []
    all_labels = []

    for batch in loader:
        batch = batch.to(device)
        x_in, edge_in, y_true, n_nodes, is_hetero = _unpack_io(batch)

        if is_hetero:
            out = model(x_in, edge_in)
        else:
            out = model(x_in, edge_in)

        loss = criterion(out, y_true.float())
        total_loss += loss.item() * n_nodes
        total_nodes += n_nodes

        preds = (torch.sigmoid(out) > 0.5)
        correct_pairs += (preds == y_true.bool()).sum().item()
        total_pairs   += y_true.numel()

        all_logits.append(out.detach().cpu())
        all_labels.append(y_true.detach().cpu())

    avg_loss = total_loss / max(total_nodes, 1)
    per_node_acc = correct_pairs / max(total_pairs, 1)

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)

    f1_score_per_task = compute_minority_f1_score_per_task(logits, labels)

    return avg_loss, per_node_acc, f1_score_per_task