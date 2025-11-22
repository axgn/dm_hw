"""
Starter PyTorch Lightning pipeline for a deep learning recommender on
MovieLens 25M (datasets/ml-25m/ratings.csv).

The template:
1) Loads ratings and builds integer vocabularies for users/items.
2) Splits interactions by time into train/val/test.
3) Creates sequence examples (history -> next item).
4) Trains a two-tower retrieval model with in-batch negatives.

Usage (CPU quick start):
    uv run python deeplearning/ml25m_deeprec_template.py --max-users 50000

Key flags to downsample for faster iteration:
    --max-users 50000 --max-rows 500000 --max-seq-len 50 --batch-size 1024
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


def load_and_index_ratings(
    ratings_path: Path,
    max_rows: Optional[int] = None,
    max_users: Optional[int] = None,
) -> Tuple[pd.DataFrame, int, int]:
    """
    Read ratings and map raw IDs to contiguous integers.
    max_rows and max_users let you shrink the dataset for faster runs.
    """
    df = pd.read_csv(
        ratings_path,
        usecols=["userId", "movieId", "timestamp"],
        nrows=max_rows,
        dtype={"userId": "int32", "movieId": "int32", "timestamp": "int64"},
        memory_map=True,
    )
    df = df.sort_values("timestamp", kind="mergesort")
    if max_users:
        # Keep the busiest users to preserve sequence signal.
        top_users = (
            df["userId"].value_counts()
            .head(max_users)
            .index.to_series()
        )
        df = df[df["userId"].isin(top_users)]
    user_ids = df["userId"].unique()
    item_ids = df["movieId"].unique()
    user2idx = {u: i for i, u in enumerate(user_ids)}
    # Reserve 0 for padding; items start at 1.
    item2idx = {m: i + 1 for i, m in enumerate(item_ids)}
    df["user_idx"] = df["userId"].map(user2idx)
    df["item_idx"] = df["movieId"].map(item2idx)
    return df, len(user2idx), len(item2idx)


def temporal_user_split(
    df: pd.DataFrame,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Time-based split per user: early interactions -> train, then val, then test.
    """
    parts = []
    for _, user_df in df.groupby("user_idx"):
        n = len(user_df)
        if n < 3:
            continue
        test_cut = max(1, int(n * (1 - test_ratio)))
        val_cut = max(1, int(test_cut * (1 - val_ratio)))
        train_part = user_df.iloc[:val_cut]
        val_part = user_df.iloc[val_cut:test_cut]
        test_part = user_df.iloc[test_cut:]
        parts.append((train_part, val_part, test_part))
    train = pd.concat([p[0] for p in parts], ignore_index=True)
    val = pd.concat([p[1] for p in parts], ignore_index=True)
    test = pd.concat([p[2] for p in parts], ignore_index=True)
    return train, val, test


@dataclass
class SequenceExample:
    history: List[int]
    target: int


def build_sequence_examples(
    df: pd.DataFrame,
    max_seq_len: int,
    min_seq_len: int = 2,
) -> List[SequenceExample]:
    """
    Turn per-user timelines into (history -> next item) training examples.
    """
    examples: List[SequenceExample] = []
    for _, user_df in df.groupby("user_idx"):
        items = user_df.sort_values("timestamp")["item_idx"].tolist()
        for i in range(1, len(items)):
            hist = items[max(0, i - max_seq_len - 1):i]
            if len(hist) < min_seq_len:
                continue
            examples.append(SequenceExample(history=hist, target=items[i]))
    return examples


class SequenceDataset(Dataset):
    def __init__(self, examples: List[SequenceExample], max_seq_len: int):
        self.examples = examples
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        ex = self.examples[idx]
        seq = ex.history[-self.max_seq_len :]
        pad_len = self.max_seq_len - len(seq)
        padded = ([0] * pad_len) + seq
        return (
            torch.tensor(padded, dtype=torch.long),
            torch.tensor(len(seq), dtype=torch.long),
            torch.tensor(ex.target, dtype=torch.long),
        )


def collate_batch(batch):
    seqs, lengths, targets = zip(*batch)
    return (
        torch.stack(seqs, dim=0),
        torch.stack(lengths, dim=0),
        torch.stack(targets, dim=0),
    )


class TwoTowerModel(pl.LightningModule):
    def __init__(
        self,
        num_items: int,
        max_seq_len: int,
        embed_dim: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        topk: int = 10,
        n_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.item_embed = nn.Embedding(num_items + 1, embed_dim, padding_idx=0)
        self.pos_embed = nn.Embedding(max_seq_len, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=4 * embed_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.topk = topk
        self.lr = lr
        self.weight_decay = weight_decay

    def encode_user(self, seq: torch.Tensor) -> torch.Tensor:
        # seq: [B, L] padded with 0 on the left
        B, L = seq.size()
        pos_ids = torch.arange(L, device=seq.device).unsqueeze(0).expand(B, -1)
        x = self.item_embed(seq) + self.pos_embed(pos_ids)
        pad_mask = seq == 0  # True where pad
        causal_mask = torch.triu(torch.ones(L, L, device=seq.device, dtype=torch.bool), diagonal=1)
        x = self.encoder(x, mask=causal_mask, src_key_padding_mask=pad_mask)
        # Take representation at last non-pad position
        lengths = (~pad_mask).sum(dim=1).clamp(min=1)
        last_idx = lengths - 1
        return x[torch.arange(B, device=seq.device), last_idx]

    def forward(self, seq: torch.Tensor, targets: torch.Tensor):
        user_vec = self.encode_user(seq)
        item_vec = self.item_embed(targets)
        logits = torch.matmul(user_vec, item_vec.T)
        return logits

    def training_step(self, batch, batch_idx):
        seq, _, target = batch
        logits = self(seq, target)
        labels = torch.arange(logits.size(0), device=logits.device)
        loss = F.cross_entropy(logits, labels)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        seq, _, target = batch
        logits = self(seq, target)
        labels = torch.arange(logits.size(0), device=logits.device)
        loss = F.cross_entropy(logits, labels)
        recall = (logits.topk(self.topk, dim=1).indices == labels.unsqueeze(1)).any(dim=1).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log(f"val_recall@{self.topk}", recall, prog_bar=True)

    def test_step(self, batch, batch_idx):
        seq, _, target = batch
        logits = self(seq, target)
        labels = torch.arange(logits.size(0), device=logits.device)
        recall = (logits.topk(self.topk, dim=1).indices == labels.unsqueeze(1)).any(dim=1).float().mean()
        self.log(f"test_recall@{self.topk}", recall, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )


class ML25MDataModule(pl.LightningDataModule):
    def __init__(
        self,
        ratings_path: Path,
        max_rows: Optional[int],
        max_users: Optional[int],
        max_seq_len: int,
        batch_size: int,
        num_workers: int = 0,
    ):
        super().__init__()
        self.ratings_path = ratings_path
        self.max_rows = max_rows
        self.max_users = max_users
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_items: Optional[int] = None

    def setup(self, stage: Optional[str] = None):
        df, num_users, num_items = load_and_index_ratings(
            self.ratings_path,
            max_rows=self.max_rows,
            max_users=self.max_users,
        )
        self.num_items = num_items
        train_df, val_df, test_df = temporal_user_split(df)
        train_ex = build_sequence_examples(train_df, self.max_seq_len)
        val_ex = build_sequence_examples(pd.concat([train_df, val_df]), self.max_seq_len)
        test_ex = build_sequence_examples(pd.concat([train_df, val_df, test_df]), self.max_seq_len)
        self.train_ds = SequenceDataset(train_ex, self.max_seq_len)
        self.val_ds = SequenceDataset(val_ex, self.max_seq_len)
        self.test_ds = SequenceDataset(test_ex, self.max_seq_len)
        print(
            f"[data] users={num_users} items={num_items} "
            f"train={len(train_ex)} val={len(val_ex)} test={len(test_ex)}"
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_batch,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_batch,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_batch,
        )


def parse_args():
    parser = argparse.ArgumentParser(description="MovieLens-25M deep learning starter")
    parser.add_argument(
        "--ratings-path",
        type=Path,
        # Resolve relative to repo root so it works no matter the cwd.
        default=Path(__file__).resolve().parents[1] / "datasets/ml-25m/ratings.csv",
        help="Path to ratings.csv (defaults to repo_root/datasets/ml-25m/ratings.csv)",
    )
    parser.add_argument("--max-rows", type=int, default=None, help="Optional row cap for quick runs")
    parser.add_argument("--max-users", type=int, default=50000, help="Keep the busiest N users (None for all)")
    parser.add_argument("--max-seq-len", type=int, default=50, help="History length to keep per user")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=15, help="Set >0 if running on Linux for faster I/O")
    parser.add_argument("--topk", type=int, default=10, help="Recall@K computed with in-batch negatives")
    parser.add_argument("--accelerator", type=str, default="auto", help="Set to 'cpu' to avoid GPU OOM")
    parser.add_argument("--devices", default="auto", help="Number of devices or 'auto'")
    parser.add_argument("--n-layers", type=int, default=2, help="Transformer encoder layers for SASRec user tower")
    parser.add_argument("--n-heads", type=int, default=4, help="Attention heads for SASRec user tower")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout for SASRec user tower")
    parser.add_argument("--precision", default="32", help="Set to '16-mixed' for torch AMP on GPU")
    parser.add_argument(
        "--log-every-n-steps",
        "--log_every_n_steps",
        type=int,
        default=10,
        help="Trainer logging frequency",
    )
    parser.add_argument(
        "--limit-train-batches",
        "--limit_train_batches",
        type=float,
        default=1.0,
        help="Fraction (0-1) or int for how many train batches to run",
    )
    parser.add_argument(
        "--limit-val-batches",
        "--limit_val_batches",
        type=float,
        default=1.0,
        help="Fraction (0-1) or int for how many val batches to run",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    pl.seed_everything(42, workers=True)
    data = ML25MDataModule(
        ratings_path=args.ratings_path,
        max_rows=args.max_rows,
        max_users=args.max_users,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    data.setup()
    if data.num_items is None:
        raise RuntimeError("num_items missing after data setup")
    model = TwoTowerModel(
        num_items=data.num_items,
        max_seq_len=args.max_seq_len,
        embed_dim=args.embed_dim,
        lr=args.lr,
        weight_decay=args.weight_decay,
        topk=args.topk,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
    )
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        gradient_clip_val=1.0,
        log_every_n_steps=args.log_every_n_steps,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        precision=args.precision,
    )
    trainer.fit(model, datamodule=data)
    trainer.test(model, datamodule=data)


if __name__ == "__main__":
    main()
