# utils/hetero.py
from torch_geometric.data import HeteroData
import torch

def make_bidirected_hetero(data):
    """
    Convert a homogeneous, directed Data object into a HeteroData with:
      node type: 'n'
      edge types:
        ('n','fwd','n'): original edges
        ('n','rev','n'): reversed edges
    Preserves x, y, train/val/test masks if present.
    """
    assert hasattr(data, "edge_index"), "Data must have an edge_index"

    hd = HeteroData()
    num_nodes = data.num_nodes

    # node features / labels
    hd['n'].num_nodes = num_nodes

    # Copy node feature matrix from the homogeneous PNA graph (shape: [N, F]
    if getattr(data, 'x', None) is not None:
        hd['n'].x = data.x

    # Copy node labels/targets (shape [N, C] where dim C represents the number of labels; we are doing multi-label classification)
    if getattr(data, 'y', None) is not None:
        hd['n'].y = data.y

    # common node-level masks (if we want to integrate in the future)
    for key in ['train_mask', 'val_mask', 'test_mask']:
        if getattr(data, key, None) is not None:
            hd['n'][key] = getattr(data, key)

    # edges
    ei = data.edge_index
    hd[('n','fwd','n')].edge_index = ei  # original direction
    hd[('n','rev','n')].edge_index = torch.flip(ei, dims=[0])  # swap src/ dst

    return hd
