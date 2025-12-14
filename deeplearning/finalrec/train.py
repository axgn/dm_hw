from __future__ import annotations
import argparse
import json
from pathlib import Path
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from model import ContentSASRec, ModelConfig
import os, sys
sys.path.append(os.path.dirname(__file__))
from config import CFG

class SeqDataset(Dataset):
    def __init__(self, pt_path: Path):
        obj = torch.load(pt_path, map_location="cpu")
        self.seq = obj["seq"]
        self.lengths = obj["len"]
        self.target = obj["target"]

    def __len__(self):
        return self.seq.size(0)

    def __getitem__(self, idx):
        return self.seq[idx], self.lengths[idx], self.target[idx]


def collate(batch):
    seq, lengths, target = zip(*batch)
    return torch.stack(seq, 0), torch.stack(lengths, 0), torch.stack(target, 0)


class LitRec(pl.LightningModule):
    def __init__(
        self,
        num_items: int,
        num_genres: int,
        num_tags: int,
        max_seq_len: int,
        cfg: ModelConfig,
        item_genres: torch.Tensor,
        item_tags: torch.Tensor,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["item_genres", "item_tags"])
        self.cfg = cfg
        self.model = ContentSASRec(
            num_items=num_items,
            num_genres=num_genres,
            num_tags=num_tags,
            max_seq_len=max_seq_len,
            cfg=cfg,
            item_genres=item_genres,
            item_tags=item_tags,
        )

    def training_step(self, batch, batch_idx):
        seq, lengths, target = batch
        loss = self.model.training_loss(seq, lengths, target)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        seq, lengths, target = batch
        loss = self.model.training_loss(seq, lengths, target)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        for k, v in self.model.sampled_metrics(seq, lengths, target).items():
            self.log("val_" + k, v, prog_bar=True)

    def test_step(self, batch, batch_idx):
        seq, lengths, target = batch
        for k, v in self.model.sampled_metrics(seq, lengths, target).items():
            self.log("test_" + k, v, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=CFG["paths"]["out_dir"])

    ap.add_argument("--epochs", type=int, default=CFG["train"]["epochs"])
    ap.add_argument("--batch-size", type=int, default=CFG["train"]["batch_size"])
    ap.add_argument("--num-workers", type=int, default=CFG["train"]["num_workers"])
    ap.add_argument("--accelerator", type=str, default=CFG["train"]["accelerator"])
    ap.add_argument("--devices", default=CFG["train"]["devices"])
    ap.add_argument("--precision", default=CFG["train"]["precision"])

    ap.add_argument("--embed-dim", type=int, default=CFG["train"]["embed_dim"])
    ap.add_argument("--n-layers", type=int, default=CFG["train"]["n_layers"])
    ap.add_argument("--n-heads", type=int, default=CFG["train"]["n_heads"])
    ap.add_argument("--dropout", type=float, default=CFG["train"]["dropout"])
    ap.add_argument("--lr", type=float, default=CFG["train"]["lr"])
    ap.add_argument("--weight-decay", type=float, default=CFG["train"]["weight_decay"])
    ap.add_argument("--n-neg", type=int, default=CFG["train"]["n_neg"])
    ap.add_argument("--topk", type=int, default=CFG["train"]["topk"])
    ap.add_argument("--seed", type=int, default=CFG["train"]["seed"])
    args = ap.parse_args()

    meta = json.loads((args.data_dir / "meta.json").read_text(encoding="utf-8"))
    max_seq_len = meta["config"]["max_seq_len"]
    num_items = meta["num_items"]
    num_genres = meta["num_genres"]
    num_tags = meta["num_tags"]

    item_genres = torch.load(args.data_dir / "item_genres.pt", map_location="cpu")
    item_tags = torch.load(args.data_dir / "item_tags.pt", map_location="cpu")

    cfg = ModelConfig(
        embed_dim=args.embed_dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        n_neg=args.n_neg,
        topk=args.topk,
    )

    pl.seed_everything(args.seed, workers=True)

    train_loader = DataLoader(
        SeqDataset(args.data_dir / "train.pt"),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
    )
    val_loader = DataLoader(
        SeqDataset(args.data_dir / "val.pt"),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
    )
    test_loader = DataLoader(
        SeqDataset(args.data_dir / "test.pt"),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
    )

    lit = LitRec(
        num_items=num_items,
        num_genres=num_genres,
        num_tags=num_tags,
        max_seq_len=max_seq_len,
        cfg=cfg,
        item_genres=item_genres,
        item_tags=item_tags,
    )

    tb_logger = TensorBoardLogger(save_dir="lightning_logs", name="finalrec")
    csv_logger = CSVLogger(save_dir="lightning_logs", name="finalrec_csv")

    ckpt_cb = ModelCheckpoint(
    dirpath=CFG["paths"]["lightning_root"] / "checkpoints",
    filename="best",
    monitor=f"val_ndcg@{args.topk}",
    mode="max",
    save_top_k=1,
    save_last=True,
)

    lr_cb = LearningRateMonitor(logging_interval="step")


    trainer = pl.Trainer(
    max_epochs=args.epochs,
    accelerator=args.accelerator,
    devices=args.devices,
    precision=args.precision,
    log_every_n_steps=10,
    logger=[tb_logger, csv_logger],
    callbacks=[ckpt_cb, lr_cb],
    default_root_dir=str(CFG["paths"]["lightning_root"]),
)

    trainer.fit(lit, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.test(lit, dataloaders=test_loader)


if __name__ == "__main__":
    main()
