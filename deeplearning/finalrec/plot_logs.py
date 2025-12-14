import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import os, sys
sys.path.append(os.path.dirname(__file__))
from config import CFG
from glob import glob
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics-csv", type=Path, default=CFG["paths"]["metrics_csv"])
    ap.add_argument("--out-dir", type=Path, default=Path("plots"))
    args = ap.parse_args()

    df = pd.read_csv(args.metrics_csv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    def last_per_epoch(col):
        sub = df[["epoch", col]].dropna()
        return sub.groupby("epoch")[col].last()

    to_plot = [
        ("train_loss_epoch", "train_loss_epoch.png", "Train Loss (epoch)"),
        ("val_loss", "val_loss.png", "Val Loss"),
        ("val_recall@10", "val_recall@10.png", "Val Recall@10"),
        ("val_ndcg@10", "val_ndcg@10.png", "Val NDCG@10"),
    ]

    for col, fname, title in to_plot:
        if col not in df.columns:
            continue
        s = last_per_epoch(col)
        plt.figure()
        plt.plot(s.index.values, s.values, marker="o")
        plt.xlabel("epoch")
        plt.title(title)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(args.out_dir / fname, dpi=200)
        plt.close()

    print("[done] saved to", args.out_dir)

if __name__ == "__main__":
    main()
