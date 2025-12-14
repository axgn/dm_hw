
from __future__ import annotations
import argparse
import json
import pickle
from pathlib import Path

import pandas as pd
import torch

from model import ContentSASRec, ModelConfig
import os, sys
sys.path.append(os.path.dirname(__file__))
from config import CFG
from glob import glob


@torch.no_grad()
def recommend(
    ckpt_path: Path,
    data_dir: Path,
    ratings_path: Path,
    user_id: int,
    topn: int = 10,
    min_rating: float = 4.0,
):
    meta = json.loads((data_dir / "meta.json").read_text(encoding="utf-8"))
    max_seq_len = meta["config"]["max_seq_len"]
    num_items = meta["num_items"]
    num_genres = meta["num_genres"]
    num_tags = meta["num_tags"]

    item_genres = torch.load(data_dir / "item_genres.pt", map_location="cpu")
    item_tags = torch.load(data_dir / "item_tags.pt", map_location="cpu")

    with open(data_dir / "mappings.pkl", "rb") as f:
        maps = pickle.load(f)
    user2idx = maps["user2idx"]
    idx2movie = maps["idx2movie"]
    movie2idx = maps["movie2idx"]

    if user_id not in user2idx:
        raise ValueError(f"userId={user_id} 不在 max_users 过滤后的集合里（或该用户无足够正反馈）。")

    df = pd.read_csv(
        ratings_path,
        usecols=["userId", "movieId", "rating", "timestamp"],
        dtype={"userId": "int32", "movieId": "int32", "rating": "float32", "timestamp": "int64"},
    )
    df = df[(df["userId"] == user_id) & (df["rating"] >= min_rating)]
    df = df.sort_values("timestamp")
    df = df[df["movieId"].isin(movie2idx.keys())]
    hist_items = [movie2idx[int(m)] for m in df["movieId"].tolist()]
    if len(hist_items) < 2:
        raise ValueError("该用户正反馈太少，无法构造序列。")
    hist_items = hist_items[-max_seq_len:]
    seq = ([0] * (max_seq_len - len(hist_items))) + hist_items
    seq = torch.tensor(seq, dtype=torch.long).unsqueeze(0)
    lengths = torch.tensor([len(hist_items)], dtype=torch.long)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    embed_dim = ckpt["state_dict"]["model.item_embed.weight"].shape[1]
    cfg = ModelConfig(embed_dim=embed_dim)

    model = ContentSASRec(
        num_items=num_items,
        num_genres=num_genres,
        num_tags=num_tags,
        max_seq_len=max_seq_len,
        cfg=cfg,
        item_genres=item_genres,
        item_tags=item_tags,
    )
    sd = {k.replace("model.", "", 1): v for k, v in ckpt["state_dict"].items() if k.startswith("model.")}
    model.load_state_dict(sd, strict=True)
    model.eval()

    u = model.encode_user(seq, lengths)
    all_item_ids = torch.arange(1, num_items + 1, dtype=torch.long).view(1, -1)
    scores = model.score(u, all_item_ids).squeeze(0)

    seen = set(hist_items)
    for it in seen:
        if 1 <= it <= num_items:
            scores[it - 1] = -1e9

    topk = torch.topk(scores, k=topn).indices.tolist()
    rec_item_idx = [i + 1 for i in topk]
    rec_movie_ids = [idx2movie[i] for i in rec_item_idx]

    item_table = pd.read_csv(data_dir / "item_table.csv")
    show = item_table[item_table["movieId"].isin(rec_movie_ids)].copy()
    show["rank"] = show["movieId"].apply(lambda x: rec_movie_ids.index(int(x)) + 1)
    show = show.sort_values("rank")[["rank", "movieId", "title", "genres"]]
    return show


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=CFG["paths"]["ckpt"])   # None/auto
    ap.add_argument("--data-dir", type=Path, default=CFG["paths"]["out_dir"])
    ap.add_argument("--ratings-path", type=Path, default=CFG["paths"]["ratings"])
    ap.add_argument("--user-id", type=int, default=CFG["infer"]["user_id"])
    ap.add_argument("--topn", type=int, default=CFG["infer"]["topn"])
    ap.add_argument("--min-rating", type=float, default=CFG["infer"]["min_rating"])

    args = ap.parse_args()
    if args.ckpt is None or not args.ckpt.exists():
        raise FileNotFoundError(f"ckpt 路径不存在：{args.ckpt}（请检查 config.py）")

    rec = recommend(args.ckpt, args.data_dir, args.ratings_path, args.user_id, args.topn, args.min_rating)
    print(rec.to_string(index=False))


if __name__ == "__main__":
    main()
