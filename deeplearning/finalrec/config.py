from pathlib import Path
CFG = {
    "paths": {
        "ratings": Path("datasets/ml-1m/ratings.csv"),
        "movies":  Path("datasets/ml-1m/movies.csv"),
        "tags":    Path("datasets/ml-1m/tags.csv"),
        
        "out_dir": Path("datasets/ml-1m/preprocessed_tensor_v2"),
        "ckpt": Path("lightning_logs/checkpoints/best.ckpt"), # 训练最好的模型路径
        "metrics_csv": Path("lightning_logs/finalrec_csv/version_2/metrics.csv"), # 训练过程中的指标记录路径 改version版本即可
        "lightning_root": Path("lightning_logs"),
    },

    "preprocess": {
        "max_rows": 200000,
        "max_users": 5000,
        "min_rating": 4.0,
        "max_seq_len": 50,
        "val_ratio": 0.1,
        "test_ratio": 0.1,
        "max_genres": 6,
        "max_tags": 10,
        "max_tag_vocab": 50000,
    },

    "train": {
        "epochs": 5,
        "batch_size": 512,
        "num_workers": 4,
        "accelerator": "auto",
        "devices": "auto",
        "precision": "32",

        "embed_dim": 64,
        "n_layers": 2,
        "n_heads": 4,
        "dropout": 0.1,
        "lr": 1e-3,
        "weight_decay": 1e-5,
        "n_neg": 500,  # 负样本数量
        "topk": 10,  #训练时的topk值
        "seed": 42,
    },

    "infer": {
        "user_id": 1, # 想要查询的user id
        "topn": 20, # 选取这个用户的topn个推荐物品
        "min_rating": 4.0,  # 只推荐评分>=min_rating的物品
    }
}
