#!/usr/bin/env python3
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import PNAConv, BatchNorm

__all__ = ["PNANetIterativeReverseMP"]

class PNANetIterativeReverseMP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        deg_forward,                # degree histogram tensor for forward edges
        deg_backward,               # degree histogram tensor for reversed edges
        num_layers: int = 6,        # use 6 convolution layers to capture the neighborhood for 6-cycles
        dropout: float = 0.1,
        ego_dim:int = 0,            # pass ego-ID dimension
        aggregators=None,
        scalers=None,
        towers: int = 4,
        pre_layers: int = 1,
        post_layers: int = 1,
        divide_input: bool = False,
        direction_schedule=None,    # I will use ["forward"]*3 + ["backward"]*3 to capture the neighborhood
    ):
        super().__init__()

        if direction_schedule is None:
            k = num_layers // 2
            direction_schedule = (["forward"] * k) + (["backward"] * (num_layers - k))
        assert len(direction_schedule) == num_layers, "direction_schedule length must equal num_layers"
        self.direction_schedule = direction_schedule

        if aggregators is None:
            aggregators = ["mean", "min", "max", "std"]
        if scalers is None:
            scalers = ["amplification", "attenuation", "identity"]

        self.ego_dim = int(ego_dim)
        self.input = nn.Linear(in_dim + self.ego_dim, hidden_dim)

        # num_layers of the form: PNAConv -> BN -> ReLU -> Dropout 
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            # Select the convolution direction for this layer
            use_deg = deg_forward if direction_schedule[i] == "forward" else deg_backward
            self.convs.append(
                PNAConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    aggregators=aggregators,
                    scalers=scalers,
                    deg=use_deg,
                    towers=towers,
                    pre_layers=pre_layers,
                    post_layers=post_layers,
                    divide_input=divide_input,
                )
            )

        self.bns = nn.ModuleList([BatchNorm(hidden_dim) for _ in range(num_layers)])
        self.dropout = dropout

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim) # define per-node logits (because my labels are per-node - so I want to predict per-node labels)
        )
        

    def forward(self, x, edge_index, edge_index_rev=None, batch=None):
        if edge_index_rev is None:
            # swap rows: [2, E] -> reversed direction
            edge_index_rev = edge_index[[1, 0], :]

        x = F.relu(self.input(x))
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            edge_idx = edge_index if self.direction_schedule[i] == "forward" else edge_index_rev
            x = conv(x, edge_idx)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.mlp(x)
    
    