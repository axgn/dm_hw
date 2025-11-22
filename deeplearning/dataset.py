import gc
import pickle
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset


# ----------------------
# 数据预处理阶段（直接生成 tensor）
# ----------------------
def preprocess_movielens_tensors(
    ratings_path: Path,
    out_dir: Path,
    max_users: Optional[int] = None,
    max_rows: Optional[int] = None,
    max_seq_len: int = 50,
):
    """
    分阶段处理 MovieLens 25M 数据：
    - 直接生成 PyTorch tensor 数据集
    - 保存为 pickle 文件
    """
    df = pd.read_csv(
        ratings_path,
        usecols=["userId", "movieId", "timestamp"],
        nrows=max_rows,
        dtype={"userId": "int32", "movieId": "int32", "timestamp": "int64"},
    )
    df = df.sort_values("timestamp")

    if max_users:
        top_users = df["userId"].value_counts().head(max_users).index
        df = df[df["userId"].isin(top_users)]

    user2idx = {u: i for i, u in enumerate(df["userId"].unique())}
    item2idx = {m: i + 1 for i, m in enumerate(df["movieId"].unique())}  # 0 用于 padding
    df["user_idx"] = df["userId"].map(user2idx)
    df["item_idx"] = df["movieId"].map(item2idx)

    # 构建 train/val/test tensors
    train_hist, train_target = [], []
    val_hist, val_target = [], []
    test_hist, test_target = [], []

    for _, user_df in df.groupby("user_idx"):
        items = user_df.sort_values("timestamp")["item_idx"].tolist()
        if len(items) < 3:
            continue
        n = len(items)
        test_cut = max(1, int(n * 0.9))
        val_cut = max(1, int(test_cut * 0.9))
        for i in range(1, n):
            hist = items[max(0, i - max_seq_len) : i]
            if len(hist) < 2:
                continue
            padded = [0] * (max_seq_len - len(hist)) + hist
            target = items[i]
            if i < val_cut:
                train_hist.append(padded)
                train_target.append(target)
            elif i < test_cut:
                val_hist.append(padded)
                val_target.append(target)
            else:
                test_hist.append(padded)
                test_target.append(target)

    # 转为 tensor
    train_ds = TensorDataset(torch.tensor(train_hist, dtype=torch.long), torch.tensor(train_target, dtype=torch.long))
    val_ds = TensorDataset(torch.tensor(val_hist, dtype=torch.long), torch.tensor(val_target, dtype=torch.long))
    test_ds = TensorDataset(torch.tensor(test_hist, dtype=torch.long), torch.tensor(test_target, dtype=torch.long))

    # 保存
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "train_tensor.pkl", "wb") as f:
        pickle.dump(train_ds, f)
    with open(out_dir / "val_tensor.pkl", "wb") as f:
        pickle.dump(val_ds, f)
    with open(out_dir / "test_tensor.pkl", "wb") as f:
        pickle.dump(test_ds, f)

    del df, train_hist, train_target, val_hist, val_target, test_hist, test_target
    gc.collect()
    return len(user2idx), len(item2idx)


# ----------------------
# 加载 TensorDataset
# ----------------------
def load_tensor_dataset(data_dir: Path, batch_size: int = 1024):
    with open(data_dir / "train_tensor.pkl", "rb") as f:
        train_ds = pickle.load(f)
    with open(data_dir / "val_tensor.pkl", "rb") as f:
        val_ds = pickle.load(f)
    with open(data_dir / "test_tensor.pkl", "rb") as f:
        test_ds = pickle.load(f)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


# ----------------------
# 使用示例
# ----------------------
if __name__ == "__main__":
    ratings_csv = Path("datasets/ml-25m/ratings.csv")
    out_dir = Path("datasets/ml-25m/preprocessed_tensor")
    max_seq_len = 50

    num_users, num_items = preprocess_movielens_tensors(ratings_csv, out_dir, max_users=50000)
    print(f"num_users={num_users}, num_items={num_items}")

    train_loader, val_loader, test_loader = load_tensor_dataset(out_dir)
    for seqs, targets in train_loader:
        print(seqs.shape, targets.shape)
        break
