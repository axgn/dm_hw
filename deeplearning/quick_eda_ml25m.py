"""
Quick EDA for MovieLens-25M without loading the whole dataset into RAM.
Computes basic counts/stats in a chunked fashion.
"""

from pathlib import Path
import pandas as pd


def main():
    ratings_path = Path(__file__).resolve().parents[1] / "datasets/ml-25m/ratings.csv"
    tags_path = Path(__file__).resolve().parents[1] / "datasets/ml-25m/tags.csv"

    chunksize = 1_000_000
    user_ids = set()
    item_ids = set()
    row_count = 0
    rating_sum = 0.0
    rating_sumsq = 0.0
    rating_min = float("inf")
    rating_max = float("-inf")
    ts_min = float("inf")
    ts_max = float("-inf")

    for chunk in pd.read_csv(
        ratings_path,
        usecols=["userId", "movieId", "rating", "timestamp"],
        dtype={
            "userId": "int32",
            "movieId": "int32",
            "rating": "float32",
            "timestamp": "int64",
        },
        chunksize=chunksize,
    ):
        row_count += len(chunk)
        user_ids.update(chunk["userId"].unique())
        item_ids.update(chunk["movieId"].unique())
        vals = chunk["rating"].values
        rating_sum += float(vals.sum())
        rating_sumsq += float((vals**2).sum())
        rating_min = min(rating_min, float(vals.min()))
        rating_max = max(rating_max, float(vals.max()))
        ts_vals = chunk["timestamp"].values
        ts_min = min(ts_min, int(ts_vals.min()))
        ts_max = max(ts_max, int(ts_vals.max()))

    mean = rating_sum / row_count
    var = rating_sumsq / row_count - mean**2
    std = var**0.5

    print("ratings.csv summary:")
    print(f"  rows={row_count:,}")
    print(f"  users={len(user_ids):,}, movies={len(item_ids):,}")
    print(f"  rating min/max={rating_min} / {rating_max}")
    print(f"  rating mean/std={mean:.4f} / {std:.4f}")
    print(f"  timestamp range={ts_min} - {ts_max}")

    head = pd.read_csv(ratings_path, nrows=3)
    print("\nfirst 3 ratings rows:")
    print(head)

    try:
        tag_df = pd.read_csv(tags_path, nrows=3)
        tag_total = sum(1 for _ in open(tags_path, "r", encoding="utf-8")) - 1
        print("\ntags.csv:")
        print(f"  rows≈{tag_total:,}")
        print("  first 3 rows:")
        print(tag_df.head(3))
    except FileNotFoundError:
        print("\ntags.csv not found")


if __name__ == "__main__":
    main()
