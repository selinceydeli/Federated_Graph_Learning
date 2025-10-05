import pandas as pd
from pathlib import Path

def compute_label_percentages(
    input_csv = "./data/y_sums.csv",
    output_csv = "./data/label_percentages.csv",
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


if __name__ == "__main__":
    compute_label_percentages()