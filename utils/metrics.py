#!/usr/bin/env python3
import os, csv
from datetime import datetime
import pandas as pd
from pathlib import Path


def compute_label_percentages(
    input_csv = "./data/y_sums.csv",
    output_csv = "./results/metrics/label_percentages.csv",
    add_mean = True
):
    '''
    Read label totals from `input_csv`compute per-split 
    percentages for each task, optionally append
    a 'mean_over_splits' row, and save to CSV.
    '''
    df = pd.read_csv(input_csv)
    if "total" not in df.columns:
        raise ValueError("Input CSV must include a 'total' column.")

    task_cols = [c for c in df.columns if c != "total"]
    if not task_cols:
        raise ValueError("No task columns found (columns other than 'total').")

    # per-split percentages
    pct = (df[task_cols].div(df["total"], axis=0) * 100.0).round(2)
    pct.index = [f"split_{i+1}" for i in range(len(pct))]

    if add_mean:
        pct.loc["mean_over_splits"] = pct.mean(axis=0).round(2)

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    pct.to_csv(output_csv, index=True)

    return pct


def append_f1_score_to_csv(
    out_csv: str,
    tasks: list[str],
    mean_f1,  
    std_f1,   
    macro_mean_percent: float,
    seeds: list[int],
    model_name: str = "PNA baseline",
):
    """
    Append a single row with mean/std per task (in %), macro mean (in %), and metadata.
    Creates the CSV (with header) if it doesn't exist.
    """
    # Ensure directory exists (if any)
    dir_ = os.path.dirname(out_csv)
    if dir_:
        os.makedirs(dir_, exist_ok=True)

    # Build header
    mean_cols = [f"{t}_mean_pct" for t in tasks]
    std_cols  = [f"{t}_std_pct"  for t in tasks]
    header = (
        ["timestamp_iso", "model", "n_runs", "seeds", "macro_mean_pct"]
        + mean_cols + std_cols
    )

    # Prepare row values
    mean_pct = (mean_f1 * 100.0).tolist()
    std_pct  = (std_f1  * 100.0).tolist()

    row = {
        "timestamp_iso": datetime.now().isoformat(timespec="seconds"),
        "model": model_name,
        "n_runs": len(seeds),
        "seeds": ",".join(map(str, seeds)),
        "macro_mean_pct": round(macro_mean_percent, 2),
        **{c: round(v, 2) for c, v in zip(mean_cols, mean_pct)},
        **{c: round(v, 2) for c, v in zip(std_cols,  std_pct)},
    }

    # Write (create header if file doesn't exist)
    file_exists = os.path.exists(out_csv)
    with open(out_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if not file_exists:
            w.writeheader()
        w.writerow(row)


def start_epoch_csv(model_name: str,
                    seed: int,
                    tasks: list,
                    out_dir: str = "./results/metrics/epoch_logs") -> str:
    """
    Creates a timestamped CSV for per-epoch metrics and writes the header.
    Returns the full path to the CSV file.
    """
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    fname = f"{ts}_{model_name}_seed{seed}.csv"
    path = os.path.join(out_dir, fname)

    header = ["epoch", "train_loss", "val_loss", "val_macro_minF1"] + [f"val_{t}_minF1" for t in tasks]

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([f"# model_name={model_name}"])
        w.writerow([f"# seed={seed}"])
        w.writerow([f"# started_at={ts}"])
        w.writerow(header)

    return path


def append_epoch_csv(csv_path: str,
                     epoch: int,
                     train_loss: float,
                     val_loss: float,
                     val_f1_tensor) -> None:
    """
    Appends a single epoch row to the CSV. val_f1_tensor is shape [num_tasks].
    """
    if hasattr(val_f1_tensor, "detach"):
        vals = val_f1_tensor.detach().cpu().tolist()
    else:
        vals = list(val_f1_tensor)
    macro = float(sum(vals) / len(vals))

    row = [int(epoch), float(train_loss), float(val_loss), float(macro)] + [float(v) for v in vals]

    with open(csv_path, "a", newline="") as f:
        csv.writer(f).writerow(row)