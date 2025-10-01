#!/usr/bin/env python3
import os
import logging
import torch
import numpy as np

from simulator import (
    GraphSimulator,
    deg_in, deg_out, fan_in, fan_out,
    Cn_check, SG2_check, BP2_check,   
)
from utils.gcn_utils import GraphData  

# Print logs on the terminal screen
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")

def define_subtasks_and_thresholds():
    """
    Define subtasks and thresholds based on the reported numbers in Table 3 of the original paper.

    Order of subtasks below defines the column order of output y:
      0  deg_in (>3)
      1  deg_out (>3)
      2  fan_in (>3)
      3  fan_out (>3)
      4  C2  (directed 2-cycle)
      5  C3
      6  C4
      7  C5
      8  C6
      9  Scatter-Gather 
      10 Biclique 
    """
    functions = [
        deg_in, deg_out, fan_in, fan_out,
        Cn_check(2), Cn_check(3), Cn_check(4), Cn_check(5), Cn_check(6),
        SG2_check,
        BP2_check,
    ]

    # Paper thresholds: degree and fan tasks have a fixed threshold of >3
    # Cycle tasks are binary (they either exist or don't exist)
    thresholds = [3, 3, 3, 3, None, None, None, None, None, None, None]
    func_names = [
        "deg_in>3", "deg_out>3", "fan_in>3", "fan_out>3",
        "cycle2", "cycle3", "cycle4", "cycle5", "cycle6",
        "scatter_gather", "biclique",
    ]
    return functions, thresholds, func_names


def set_y_with_labels(funcs, thresh, data: GraphData):
    data.set_y(funcs, thresh)
    return data


def write_label_stats(path, names, datasets):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(",".join(names) + ",total\n")
        for d in datasets:
            # y is [num_nodes, num_tasks] of 0/1 floats; sum per task:
            sums = list(torch.sum(d.y, dim=0).long().cpu().numpy().tolist())
            f.write(",".join(str(x) for x in sums) + f",{d.y.shape[0]}\n")


def main():
    # Below parameters are defined based on Appendix D.2 of the original paper
    n = 8192        # number of nodes
    d = 6           # average degree
    r = 11.1        # radius
    num_graphs = 1  # one connected component generator call per data split to prevent data leakeage
    generator = "chordal"  # describes the random-circulant-like generator mentioned in the paper (to my understanding)
    bidirectional = False  # have a directed multigraph (needed for directed cycles)
    max_time = n    # TODO: defined this parameter arbitrarily (not sure what it means)

    # Build simulator once per split to ensure independent graphs, as described in the paper
    def make_sim():
        return GraphSimulator(
            num_nodes=n,
            avg_degree=d,
            num_edges=None,
            max_time=max_time,
            network_type="type1",
            readout="node",
            node_feats=False,
            bidirectional=bidirectional,
            delta=r,                
            num_graphs=num_graphs,
            generator=generator,    
        )

    logging.info("Generating train/val/test graphs (independent random circulant graphs).")
    tr = make_sim().generate_pytorch_graph()
    va = make_sim().generate_pytorch_graph()
    te = make_sim().generate_pytorch_graph()

    functions, thresholds, names = define_subtasks_and_thresholds()

    logging.info("Computing labels with paper thresholds for train/val/test splits.")
    tr = set_y_with_labels(functions, thresholds, tr)
    va = set_y_with_labels(functions, thresholds, va)
    te = set_y_with_labels(functions, thresholds, te)

    out_dir = "./data"
    os.makedirs(out_dir, exist_ok=True)

    # Log label stats for sanity check
    write_label_stats(os.path.join(out_dir, "y_sums.csv"), names, [tr, va, te])
    logging.info("Wrote label totals to %s", os.path.join(out_dir, "y_sums.csv"))
    
    torch.save(tr, os.path.join(out_dir, "train.pt"))
    torch.save(va, os.path.join(out_dir, "val.pt"))
    torch.save(te, os.path.join(out_dir, "test.pt"))
    logging.info("Saved train/val/test GraphData objects.")


if __name__ == "__main__":
    main()
