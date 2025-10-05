import os, csv
from datetime import datetime

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