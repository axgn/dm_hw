

from __future__ import annotations

import argparse
import datetime
import json
import pickle
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
import os, sys
sys.path.append(os.path.dirname(__file__))
from config import CFG


@dataclass
class PreprocessConfig:
    max_rows: Optional[int] = None
    max_users: Optional[int] = 50000
    min_rating: float = 4.0                 # 将评分 >= min_rating 视作正反馈
    max_seq_len: int = 50                   # 历史序列长度（左侧padding）
    min_seq_len: int = 2                    # 至少这么长才构造样本
    val_ratio: float = 0.1                  # 按用户时间序列切分
    test_ratio: float = 0.1
    max_genres: int = 6                     # 每部电影最多保留多少个genre
    max_tags: int = 10                      # 每部电影最多保留多少个tag
    max_tag_vocab: int = 50000              # tag词表上限（按频率截断）


def read_ratings(ratings_path: Path, cfg: PreprocessConfig) -> pd.DataFrame:
    df = pd.read_csv(
        ratings_path,
        usecols=["userId", "movieId", "rating", "timestamp"],
        nrows=cfg.max_rows,
        dtype={"userId": "int32", "movieId": "int32", "rating": "float32", "timestamp": "int64"},
        memory_map=True,
    )
    df = df[df["rating"] >= cfg.min_rating]
    df = df.sort_values("timestamp", kind="mergesort")  # stable sort
    if cfg.max_users:
        top_users = df["userId"].value_counts().head(cfg.max_users).index
        df = df[df["userId"].isin(top_users)]
    return df


def build_id_maps(df: pd.DataFrame) -> Tuple[Dict[int, int], Dict[int, int], Dict[int, int], Dict[int, int]]:
    user_ids = df["userId"].unique()
    movie_ids = df["movieId"].unique()
    user2idx = {int(u): i for i, u in enumerate(user_ids)}
    idx2user = {i: int(u) for i, u in enumerate(user_ids)}
    movie2idx = {int(m): i + 1 for i, m in enumerate(movie_ids)}  # 0 reserved for padding
    idx2movie = {i + 1: int(m) for i, m in enumerate(movie_ids)}
    return user2idx, idx2user, movie2idx, idx2movie


def read_movies(movies_path: Path, movie2idx: Dict[int, int]) -> pd.DataFrame:
    mv = pd.read_csv(
        movies_path,
        usecols=["movieId", "title", "genres"],
        dtype={"movieId": "int32", "title": "string", "genres": "string"},
    )
    mv = mv[mv["movieId"].isin(movie2idx.keys())].copy()
    return mv


def build_genre_tensor(
    movies_df: pd.DataFrame, movie2idx: Dict[int, int], max_genres: int
) -> Tuple[torch.Tensor, Dict[str, int]]:
    genre_counter = Counter()
    movie_genres: Dict[int, List[str]] = {}
    for _, r in movies_df.iterrows():
        gs = str(r["genres"]).split("|") if pd.notna(r["genres"]) else []
        gs = [g for g in gs if g and g != "(no genres listed)"]
        movie_genres[int(r["movieId"])] = gs[:max_genres]
        genre_counter.update(gs)

    genres_sorted = [g for g, _ in genre_counter.most_common()]
    genre2idx = {g: i + 1 for i, g in enumerate(genres_sorted)}  # 0 padding

    num_items = max(movie2idx.values())
    out = torch.zeros((num_items + 1, max_genres), dtype=torch.long)
    for movie_id, gs in movie_genres.items():
        item_idx = movie2idx[movie_id]
        ids = [genre2idx[g] for g in gs if g in genre2idx][:max_genres]
        if ids:
            out[item_idx, : len(ids)] = torch.tensor(ids, dtype=torch.long)
    return out, genre2idx


def read_tags(tags_path: Path, movie2idx: Dict[int, int]) -> Optional[pd.DataFrame]:
    if not tags_path.exists():
        return None
    tg = pd.read_csv(
        tags_path,
        usecols=["movieId", "tag"],
        dtype={"movieId": "int32", "tag": "string"},
    )
    tg = tg[tg["movieId"].isin(movie2idx.keys())].copy()
    tg["tag"] = tg["tag"].astype("string").str.lower().str.strip()
    tg = tg[tg["tag"].notna() & (tg["tag"].str.len() > 0)]
    return tg


def build_tag_tensor(
    tags_df: Optional[pd.DataFrame], movie2idx: Dict[int, int], max_tags: int, max_tag_vocab: int
) -> Tuple[torch.Tensor, Dict[str, int]]:
    num_items = max(movie2idx.values())
    out = torch.zeros((num_items + 1, max_tags), dtype=torch.long)
    if tags_df is None or len(tags_df) == 0:
        return out, {}

    tag_counter = Counter(tags_df["tag"].tolist())
    vocab = [t for t, _ in tag_counter.most_common(max_tag_vocab)]
    tag2idx = {t: i + 1 for i, t in enumerate(vocab)}  # 0 padding

    per_movie = defaultdict(Counter)
    for movie_id, tag in zip(tags_df["movieId"].tolist(), tags_df["tag"].tolist()):
        if tag in tag2idx:
            per_movie[int(movie_id)][tag] += 1

    for movie_id, ctr in per_movie.items():
        item_idx = movie2idx[int(movie_id)]
        top_tags = [t for t, _ in ctr.most_common(max_tags)]
        ids = [tag2idx[t] for t in top_tags if t in tag2idx][:max_tags]
        if ids:
            out[item_idx, : len(ids)] = torch.tensor(ids, dtype=torch.long)
    return out, tag2idx


def temporal_user_split(items: List[int], val_ratio: float, test_ratio: float) -> Tuple[List[int], List[int], List[int]]:
    n = len(items)
    if n < 3:
        return [], [], []
    test_cut = max(1, int(n * (1 - test_ratio)))
    val_cut = max(1, int(test_cut * (1 - val_ratio)))
    return items[:val_cut], items[val_cut:test_cut], items[test_cut:]


def build_examples(user2items: Dict[int, List[int]], cfg: PreprocessConfig, split: str) -> Dict[str, torch.Tensor]:
    seqs, lens, targets = [], [], []
    for _, items in user2items.items():
        if len(items) < 3:
            continue

        tr, va, te = temporal_user_split(items, cfg.val_ratio, cfg.test_ratio)

        if split == "train":
            source = tr
            start_i = 1
            end_i = len(tr)                 # 只预测 train 段内部 next-item
        elif split == "val":
            source = tr + va
            start_i = len(tr)               # 只预测 va 段的目标
            end_i = len(tr) + len(va)
        elif split == "test":
            source = tr + va + te
            start_i = len(tr) + len(va)     # 只预测 te 段的目标
            end_i = len(source)
        else:
            raise ValueError(split)

        if end_i <= start_i:
            continue

        for i in range(start_i, end_i):
            hist = source[max(0, i - cfg.max_seq_len): i]
            if len(hist) < cfg.min_seq_len:
                continue
            pad_len = cfg.max_seq_len - len(hist)
            seqs.append(([0] * pad_len) + hist)
            lens.append(len(hist))
            targets.append(source[i])

    if len(seqs) == 0:
        raise RuntimeError(f"No examples built for split={split}. Check filters/max_users/min_rating.")
    return {
        "seq": torch.tensor(seqs, dtype=torch.long),
        "len": torch.tensor(lens, dtype=torch.long),
        "target": torch.tensor(targets, dtype=torch.long),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratings-path", type=Path, default=CFG["paths"]["ratings"])
    ap.add_argument("--movies-path",  type=Path, default=CFG["paths"]["movies"])
    ap.add_argument("--tags-path",    type=Path, default=CFG["paths"]["tags"])
    ap.add_argument("--out-dir",      type=Path, default=CFG["paths"]["out_dir"])
    ap.add_argument("--max-rows", type=int, default=CFG["preprocess"]["max_rows"])
    ap.add_argument("--max-users", type=int, default=CFG["preprocess"]["max_users"])
    ap.add_argument("--min-rating", type=float, default=CFG["preprocess"]["min_rating"])
    ap.add_argument("--max-seq-len", type=int, default=CFG["preprocess"]["max_seq_len"])
    ap.add_argument("--val-ratio", type=float, default=CFG["preprocess"]["val_ratio"])
    ap.add_argument("--test-ratio", type=float, default=CFG["preprocess"]["test_ratio"])
    ap.add_argument("--max-genres", type=int, default=CFG["preprocess"]["max_genres"])
    ap.add_argument("--max-tags", type=int, default=CFG["preprocess"]["max_tags"])
    ap.add_argument("--max-tag-vocab", type=int, default=CFG["preprocess"]["max_tag_vocab"])

    args = ap.parse_args()
    cfg = PreprocessConfig(
        max_rows=args.max_rows,
        max_users=args.max_users,
        min_rating=args.min_rating,
        max_seq_len=args.max_seq_len,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        max_genres=args.max_genres,
        max_tags=args.max_tags,
        max_tag_vocab=args.max_tag_vocab,
    )

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    ratings = read_ratings(args.ratings_path, cfg)
    user2idx, idx2user, movie2idx, idx2movie = build_id_maps(ratings)
    ratings["user_idx"] = ratings["userId"].map(user2idx)
    ratings["item_idx"] = ratings["movieId"].map(movie2idx)

    user2items: Dict[int, List[int]] = {}
    for u, udf in ratings.groupby("user_idx"):
        user2items[int(u)] = udf.sort_values("timestamp")["item_idx"].tolist()

    movies_df = read_movies(args.movies_path, movie2idx)
    genre_tensor, genre2idx = build_genre_tensor(movies_df, movie2idx, cfg.max_genres)

    tags_path = Path(args.tags_path) if args.tags_path is not None else (args.movies_path.parent / "tags.csv")
    tags_df = read_tags(tags_path, movie2idx)
    tag_tensor, tag2idx = build_tag_tensor(tags_df, movie2idx, cfg.max_tags, cfg.max_tag_vocab)

    train = build_examples(user2items, cfg, "train")
    val = build_examples(user2items, cfg, "val")
    test = build_examples(user2items, cfg, "test")

    torch.save(train, out_dir / "train.pt")
    torch.save(val, out_dir / "val.pt")
    torch.save(test, out_dir / "test.pt")
    torch.save(genre_tensor, out_dir / "item_genres.pt")
    torch.save(tag_tensor, out_dir / "item_tags.pt")

    with open(out_dir / "mappings.pkl", "wb") as f:
        pickle.dump(
            {
                "user2idx": user2idx,
                "idx2user": idx2user,
                "movie2idx": movie2idx,
                "idx2movie": idx2movie,
                "genre2idx": genre2idx,
                "tag2idx": tag2idx,
            },
            f,
        )

    item_table = movies_df.copy()
    item_table["item_idx"] = item_table["movieId"].map(movie2idx).astype("int64")
    item_table = item_table[["item_idx", "movieId", "title", "genres"]].sort_values("item_idx")
    item_table.to_csv(out_dir / "item_table.csv", index=False)

    meta = {
        "created_at": datetime.datetime.utcnow().isoformat() + "Z",
        "config": asdict(cfg),
        "num_users": len(user2idx),
        "num_items": len(movie2idx),
        "num_genres": len(genre2idx),
        "num_tags": len(tag2idx),
        "paths": {
            "train": "train.pt",
            "val": "val.pt",
            "test": "test.pt",
            "item_genres": "item_genres.pt",
            "item_tags": "item_tags.pt",
            "mappings": "mappings.pkl",
            "item_table": "item_table.csv",
        },
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[done] out_dir =", out_dir)
    print(f"users={meta['num_users']:,} items={meta['num_items']:,} genres={meta['num_genres']:,} tags={meta['num_tags']:,}")
    print(f"train={train['seq'].shape[0]:,} val={val['seq'].shape[0]:,} test={test['seq'].shape[0]:,}")


if __name__ == "__main__":
    main()
