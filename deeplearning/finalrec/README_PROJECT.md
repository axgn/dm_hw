## 0）快速开始
首先  
在datasets目录下根据ml-25m的目录结构下载一个ml-1m的数据集

然后直接运行以下命令即可：
我把参数都集合到config.py了，想要改照着看就行了。
## 1）预处理
```bash
python preprocess_ml25m.py   
```

## 2）训练
```bash
python train.py  
```

## 3）推理
```bash
python infer.py 
```
可以选择config更改或者直接在命令行更改想要预测的用户id
## 4） 推理
```bash
python plot_logs.py 
```
在plot文件夹生成四幅图，到时候那张val_loss 不用汇报 ，其他三张图汇报即可

## 5）
- 问题定义：隐式反馈序列推荐（next-item prediction）
- 特征：itemId + genres + tags
- 模型：Content-aware SASRec（Transformer encoder）

！！！目前属于sampled / in-batch推荐，不是对整个电影样本推荐
    在 1 个正确答案 + n_neg(比如说200) 个随机干扰项里，模型能不能把正确答案排进前 topk(比如说10)
- 训练：sampled softmax（每个正样本采样 n_neg 个负例）
- 指标：sampled Recall@K / NDCG@K；可选 full ranking

如果后续先搞全库评估思路是：对每个 test 样本，把模型对 全部 item 打分，然后取 top10，再算 recall/ndcg。
对每个样本得到用户向量 u
预先拿到全部 item 向量矩阵 V（shape: [num_items, dim]）
scores = u @ V.T
把用户历史里出现过的 item 分数设为 -inf（避免推荐看过的）
topk = argsort(scores)[:10]
看真实 target 是否在 topk

将评分>=4 视为正反馈；低分/未交互视为负反馈（隐式化）。
