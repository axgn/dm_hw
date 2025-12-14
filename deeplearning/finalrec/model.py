
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    embed_dim: int = 64
    n_layers: int = 2
    n_heads: int = 4
    dropout: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 1e-5
    n_neg: int = 50
    topk: int = 10


class ContentSASRec(nn.Module):
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
        self.num_items = num_items
        self.num_genres = num_genres
        self.num_tags = num_tags
        self.max_seq_len = max_seq_len
        self.cfg = cfg

        self.item_embed = nn.Embedding(num_items + 1, cfg.embed_dim, padding_idx=0)
        self.genre_embed = nn.Embedding(num_genres + 1, cfg.embed_dim, padding_idx=0)
        self.tag_embed = nn.Embedding(num_tags + 1, cfg.embed_dim, padding_idx=0)
        self.pos_embed = nn.Embedding(max_seq_len, cfg.embed_dim)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.embed_dim,
            nhead=cfg.n_heads,
            dim_feedforward=4 * cfg.embed_dim,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.n_layers)
        self.out_norm = nn.LayerNorm(cfg.embed_dim)

        self.register_buffer("item_genres", item_genres.long(), persistent=False)
        self.register_buffer("item_tags", item_tags.long(), persistent=False)

    def item_repr(self, item_ids: torch.Tensor) -> torch.Tensor:
        x = self.item_embed(item_ids)

        g = self.item_genres[item_ids]            # [..., G]
        g_emb = self.genre_embed(g)               # [..., G, D]
        g_mask = (g != 0).unsqueeze(-1)
        g_sum = (g_emb * g_mask).sum(dim=-2)
        g_cnt = g_mask.sum(dim=-2).clamp(min=1)
        g_mean = g_sum / g_cnt

        t = self.item_tags[item_ids]              # [..., T]
        t_emb = self.tag_embed(t)                 # [..., T, D]
        t_mask = (t != 0).unsqueeze(-1)
        t_sum = (t_emb * t_mask).sum(dim=-2)
        t_cnt = t_mask.sum(dim=-2).clamp(min=1)
        t_mean = t_sum / t_cnt

        return self.out_norm(x + g_mean + t_mean)

    def encode_user(self, seq: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L = seq.shape
        pos = torch.arange(L, device=seq.device).unsqueeze(0).expand(B, -1)

        x = self.item_repr(seq) + self.pos_embed(pos)
        pad_mask = seq == 0
        causal_mask = torch.triu(torch.ones(L, L, device=seq.device, dtype=torch.bool), diagonal=1)
        x = self.encoder(x, mask=causal_mask, src_key_padding_mask=pad_mask)

        if lengths is None:
            lengths = (~pad_mask).sum(dim=1).clamp(min=1)
        last_idx = lengths - 1
        u = x[torch.arange(B, device=seq.device), last_idx]
        return self.out_norm(u)

    def score(self, user_vec: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        item_vec = self.item_repr(item_ids)       # [B, C, D]
        return torch.einsum("bd,bcd->bc", user_vec, item_vec)

    def sample_negatives(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.randint(1, self.num_items + 1, (batch_size, self.cfg.n_neg), device=device)

    def training_loss(self, seq: torch.Tensor, lengths: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        B = seq.size(0)
        u = self.encode_user(seq, lengths)
        neg = self.sample_negatives(B, seq.device)
        cand = torch.cat([pos.view(B, 1), neg], dim=1)
        logits = self.score(u, cand)
        labels = torch.zeros(B, dtype=torch.long, device=seq.device)  # pos at index 0
        return F.cross_entropy(logits, labels)

    @torch.no_grad()
    def sampled_metrics(self, seq: torch.Tensor, lengths: torch.Tensor, pos: torch.Tensor) -> dict:
        B = seq.size(0)
        u = self.encode_user(seq, lengths)
        neg = self.sample_negatives(B, seq.device)
        cand = torch.cat([pos.view(B, 1), neg], dim=1)
        logits = self.score(u, cand)

        k = min(self.cfg.topk, logits.size(1))
        topk_idx = torch.topk(logits, k=k, dim=1).indices
        hit = (topk_idx == 0).any(dim=1).float().mean()

        sorted_idx = torch.argsort(logits, dim=1, descending=True)
        pos_rank = (sorted_idx == 0).nonzero(as_tuple=False)
        ranks = torch.full((B,), fill_value=10**9, device=seq.device, dtype=torch.long)
        ranks[pos_rank[:, 0]] = pos_rank[:, 1]
        ndcg = torch.where(
            ranks < k,
            1.0 / torch.log2(ranks.float() + 2.0),
            torch.zeros_like(ranks, dtype=torch.float),
        ).mean()
        return {f"recall@{k}": hit, f"ndcg@{k}": ndcg}
