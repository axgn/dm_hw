from __future__ import annotations
import argparse
import json
from pathlib import Path
import os, sys

import torch
from torch.utils.data import Dataset, DataLoader

sys.path.append(os.path.dirname(__file__))
from config import CFG
from model import ContentSASRec, ModelConfig


class SeqDataset(Dataset):
    """读取 preprocess 产物：train.pt / val.pt / test.pt"""
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


def _mask_seen_in_chunk(scores_chunk: torch.Tensor, seq: torch.Tensor, start: int, end: int):
    """
    scores_chunk: [B, C]  对应 item_id in [start, end)
    seq: [B, L] 历史序列（含0 padding）
    把历史出现过的 item 在本 chunk 内设为 -inf
    """
    # 找出落在 [start, end) 的历史 item
    m = (seq > 0) & (seq >= start) & (seq < end)
    if not m.any():
        return
    idx = m.nonzero(as_tuple=False)  # [N,2] => (b, pos)
    b = idx[:, 0]
    item_ids = seq[m]                # [N]
    c = item_ids - start             # 映射到 chunk 内列坐标
    scores_chunk[b, c] = -1e9


@torch.no_grad()
def full_ranking_metrics(
    model: ContentSASRec,
    seq: torch.Tensor,
    lengths: torch.Tensor,
    target: torch.Tensor,
    num_items: int,
    topk: int = 10,
    chunk_size: int = 5000,
):
    """
    Full-catalog ranking:
    - 对每个样本：对所有 item 打分（chunked）
    - mask 掉历史看过的
    - 计算 Recall@K 和 NDCG@K（单个 target）
    """
    device = seq.device
    B = seq.size(0)

    u = model.encode_user(seq, lengths)  # [B, D]

    # target 的分数（用于算rank）
    t_score = model.score(u, target.view(B, 1)).squeeze(1)  # [B]

    # streaming topk
    top_scores = torch.full((B, topk), -1e9, device=device)
    top_items = torch.zeros((B, topk), dtype=torch.long, device=device)

    better_cnt = torch.zeros((B,), dtype=torch.long, device=device)

    # 分 chunk 遍历所有 item: [1..num_items]
    for start in range(1, num_items + 1, chunk_size):
        end = min(num_items + 1, start + chunk_size)
        item_ids_1d = torch.arange(start, end, device=device)      # [C]

        item_vec = model.item_repr(item_ids_1d)                    # [C, D] 只算一次
        s = u @ item_vec.T                                         # [B, C]

        # 如果你后面需要 item_ids 做 topk 合并，再“轻量扩展”即可（扩展 int 很便宜）
        item_ids = item_ids_1d.view(1, -1).expand(B, -1)           # [B, C]

        # item_ids = item_ids.view(1, -1).expand(B, -1)  # [B, C]

        s = model.score(u, item_ids)  # [B, C]
        _mask_seen_in_chunk(s, seq, start, end)

        # rank: 统计有多少 item 分数 > target 分数
        better_cnt += (s > t_score.unsqueeze(1)).sum(dim=1)

        # topk merge
        comb_scores = torch.cat([top_scores, s], dim=1)
        comb_items = torch.cat([top_items, item_ids], dim=1)
        top_scores, idx = torch.topk(comb_scores, k=topk, dim=1)
        top_items = torch.gather(comb_items, 1, idx)

    # Recall@K: target 是否在 topK
    hit = (top_items == target.unsqueeze(1)).any(dim=1).float()

    # NDCG@K: rank=better+1 (1-index)
    rank = better_cnt + 1
    ndcg = torch.where(
        rank <= topk,
        1.0 / torch.log2(rank.float() + 1.0),
        torch.zeros_like(rank, dtype=torch.float),
    )

    return hit.mean().item(), ndcg.mean().item()


def load_model_from_ckpt(ckpt_path: Path, data_dir: Path, topk: int, device: torch.device):
    meta = json.loads((data_dir / "meta.json").read_text(encoding="utf-8"))
    max_seq_len = meta["config"]["max_seq_len"]
    num_items = meta["num_items"]
    num_genres = meta["num_genres"]
    num_tags = meta["num_tags"]

    item_genres = torch.load(data_dir / "item_genres.pt", map_location="cpu")
    item_tags = torch.load(data_dir / "item_tags.pt", map_location="cpu")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    embed_dim = ckpt["state_dict"]["model.item_embed.weight"].shape[1]

    cfg = ModelConfig(
        embed_dim=embed_dim,
        n_layers=CFG["train"]["n_layers"],
        n_heads=CFG["train"]["n_heads"],
        dropout=CFG["train"]["dropout"],
        lr=CFG["train"]["lr"],
        weight_decay=CFG["train"]["weight_decay"],
        n_neg=CFG["train"]["n_neg"],
        topk=topk,
    )

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
    model.to(device).eval()
    return model, num_items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=CFG["paths"]["ckpt"])
    ap.add_argument("--data-dir", type=Path, default=CFG["paths"]["out_dir"])
    ap.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    ap.add_argument("--topk", type=int, default=CFG["train"]["topk"])
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--chunk-size", type=int, default=5000)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    pt_path = args.data_dir / f"{args.split}.pt"
    if not pt_path.exists():
        raise FileNotFoundError(f"找不到 {pt_path}（先 preprocess 生成 *.pt）")
    if args.ckpt is None or not args.ckpt.exists():
        raise FileNotFoundError(f"ckpt 路径不存在：{args.ckpt}（请检查 config.py）")

    device = torch.device(args.device)
    model, num_items = load_model_from_ckpt(args.ckpt, args.data_dir, args.topk, device)

    loader = DataLoader(
        SeqDataset(pt_path),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
    )

    recalls, ndcgs, n = 0.0, 0.0, 0
    for seq, lengths, target in loader:
        seq = seq.to(device)
        lengths = lengths.to(device)
        target = target.to(device)

        r, d = full_ranking_metrics(
            model=model,
            seq=seq,
            lengths=lengths,
            target=target,
            num_items=num_items,
            topk=args.topk,
            chunk_size=args.chunk_size,
        )
        bs = seq.size(0)
        recalls += r * bs
        ndcgs += d * bs
        n += bs

    print(f"[FULL-RANK] split={args.split}  Recall@{args.topk}={recalls/n:.6f}  NDCG@{args.topk}={ndcgs/n:.6f}")


if __name__ == "__main__":
    main()
