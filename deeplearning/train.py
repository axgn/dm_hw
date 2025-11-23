import pickle
from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# -------- 模型 --------
class TwoTower(pl.LightningModule):
    def __init__(self, num_items, max_seq_len, embed_dim=64, lr=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.item_embed = nn.Embedding(num_items + 1, embed_dim, padding_idx=0)
        self.pos_embed = nn.Embedding(max_seq_len, embed_dim)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=4,
                dim_feedforward=embed_dim * 4,
                dropout=0.1,
                batch_first=True,
                activation="gelu",
                norm_first=True,
            ),
            num_layers=2,
        )
        self.lr = lr

    def encode_user(self, seq):
        B, L = seq.size()
        pos = torch.arange(L, device=seq.device).unsqueeze(0).expand(B, -1)
        x = self.item_embed(seq) + self.pos_embed(pos)
        pad_mask = seq == 0
        causal = torch.triu(torch.ones(L, L, device=seq.device, dtype=torch.bool), 1)
        x = self.encoder(x, mask=causal, src_key_padding_mask=pad_mask)
        last = (~pad_mask).sum(1) - 1
        return x[torch.arange(B), last]

    def forward(self, seq, targets):
        u = self.encode_user(seq)
        i = self.item_embed(targets)
        return u @ i.T

    def training_step(self, batch, _):
        seq, target = batch
        logits = self(seq, target)
        labels = torch.arange(logits.size(0), device=logits.device)
        loss = F.cross_entropy(logits, labels)
        self.log("train_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        seq, target = batch
        logits = self(seq, target)
        labels = torch.arange(logits.size(0), device=logits.device)
        loss = F.cross_entropy(logits, labels)

        # top-1 正确率，因为目标必须是对应行的物品
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()

        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)

        scores = logits
        # 每行按从大到小排序

        topk = torch.topk(scores, k=10, dim=1).indices
        hits = (topk == labels.unsqueeze(1)).any(dim=1).float()
        recall10 = hits.mean()
        self.log("recall@10", recall10, prog_bar=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)


# -------- DataModule: 从 TensorDataset 文件加载 --------
class TensorDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: Path, batch_size=1024):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def setup(self, stage=None):
        # 直接加载 pickle 文件
        with open(self.data_dir / "train_tensor.pkl", "rb") as f:
            self.train_ds = pickle.load(f)
        with open(self.data_dir / "val_tensor.pkl", "rb") as f:
            self.val_ds = pickle.load(f)
        with open(self.data_dir / "test_tensor.pkl", "rb") as f:
            self.test_ds = pickle.load(f)

        print(f"[TensorDataModule] train={len(self.train_ds)}, val={len(self.val_ds)}, test={len(self.test_ds)}")

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False)


# -------- 使用示例 --------
if __name__ == "__main__":
    data_dir = Path("datasets/ml-25m/preprocessed_tensor")
    dm = TensorDataModule(data_dir)
    print("data loaded.")
    dm.setup()
    all_targets = torch.cat(
        [
            dm.train_ds.tensors[1],  # target 在 TensorDataset 中通常是第二个 tensor
            dm.val_ds.tensors[1],
            dm.test_ds.tensors[1],
        ]
    )
    max_seq_len = 50
    num_items = int(all_targets.max().item())
    model = TwoTower(num_items=num_items, max_seq_len=max_seq_len)
    trainer = pl.Trainer( max_epochs=10,
    accelerator="gpu",
    devices=1,
    precision="16-mixed",
    log_every_n_steps=10,)
    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm)
