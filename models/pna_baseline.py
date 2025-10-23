#!/usr/bin/env python3
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import PNAConv, BatchNorm

'''
The PNANet class described in this script is defined based on the following paper:
https://arxiv.org/pdf/2004.05718

The PNANet definition is adapted based on the configurations specified in the
Provably Powerful GNNs paper:
https://arxiv.org/abs/2306.11586
'''

__all__ = ["PNANet"]

class PNANet(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        deg,                    # degree histogram tensor for PNA
        num_layers: int = 6,
        dropout: float = 0.1,
        ego_dim:int = 0,        # pass ego-ID dimension
        aggregators=None,
        scalers=None,
        towers: int = 4,
        pre_layers: int = 1,
        post_layers: int = 1,
        divide_input: bool = False,
    ):
        super().__init__()

        if aggregators is None:
            aggregators = ["mean", "min", "max", "std"]
        if scalers is None:
            scalers = ["amplification", "attenuation", "identity"]

        self.ego_dim = int(ego_dim)
        self.input = nn.Linear(in_dim + self.ego_dim, hidden_dim)

        # num_layers of the form: PNAConv -> BN -> ReLU -> Dropout 
        self.convs = nn.ModuleList([
            PNAConv(
                hidden_dim,
                hidden_dim,
                aggregators,
                scalers,
                deg=deg,
                towers=towers,
                pre_layers=pre_layers,
                post_layers=post_layers,
                divide_input=divide_input,
            )
            for _ in range(num_layers)
        ])

        self.bns = nn.ModuleList([BatchNorm(hidden_dim) for _ in range(num_layers)])
        self.dropout = dropout

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim) # define per-node logits (because my labels are per-node - so I want to predict per-node labels)
        )
        

    def forward(self, x, edge_index, batch=None):
        x = F.relu(self.input(x))
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.mlp(x) 