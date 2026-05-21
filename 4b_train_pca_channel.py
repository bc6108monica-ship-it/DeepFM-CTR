"""
Step 4b: PCA 降维版双通道 DeepFM（消融实验）
用途：和 4train_dualchannel.py 对比，验证"直接输入1024维 > PCA压缩到64维"

相比 4train_dualchannel.py，只改了三处：
  [改动1] 第3节：加了 PCA fit_transform，把1024维压到64维
  [改动2] 第5节：DenseFeat dimension 从 1024 改成 64
  [改动3] 第6节：train_input/test_input 换成 PCA 压缩后的向量

其余代码（模型结构、训练参数、评估逻辑）和第4步完全一致，
保证对比公平，唯一变量就是"有没有用PCA压缩"
"""

import pandas as pd
import numpy as np
import torch
from sklearn.decomposition import PCA          # [改动1] 新增导入
from deepctr_torch.models import DeepFM
from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
from sklearn.metrics import roc_auc_score
import os

# ── 0. 路径配置 ────────────────────────────────────────────────────────────
DATA_DIR = "./data/processed"
EMB_DIR  = "./data/embeddings"
OUT_DIR  = "./results"
os.makedirs(OUT_DIR, exist_ok=True)

# ── 1. 读数据 ──────────────────────────────────────────────────────────────
train = pd.read_csv(f"{DATA_DIR}/train.csv")
test  = pd.read_csv(f"{DATA_DIR}/test.csv")
print(f"train: {train.shape}, test: {test.shape}")

# ── 2. 加载 LLM 向量矩阵 ──────────────────────────────────────────────────
emb_matrix = np.load(f"{EMB_DIR}/movie_embeddings.npy")
print(f"LLM向量矩阵: {emb_matrix.shape}")

train_movie_emb = emb_matrix[train["movie_id"].values]
test_movie_emb  = emb_matrix[test["movie_id"].values]

# ── 3. [改动1] PCA 降维：1024维 → 64维 ───────────────────────────────────
# 注意：PCA 只能在训练集上 fit，不能用测试集
# 原因和 LabelEncoder 一样：测试集模拟"未来数据"，不能提前看到
PCA_DIM = 64
pca = PCA(n_components=PCA_DIM)
train_movie_emb_pca = pca.fit_transform(train_movie_emb)   # fit+transform
test_movie_emb_pca  = pca.transform(test_movie_emb)         # 只transform，不fit

print(f"PCA压缩后: {train_movie_emb_pca.shape}")
print(f"保留方差比例: {pca.explained_variance_ratio_.sum():.4f}")
# 这个数字面试时有用：说明PCA保留了多少%的方差信息
# 通常64维能保留约70-85%，剩下的语义信息被丢弃了

# ── 4. 定义特征列 ──────────────────────────────────────────────────────────
sparse_features = ["user_id", "movie_id", "gender", "age", "occupation", "zip"]

sparse_feature_columns = [
    SparseFeat(
        name            = feat,
        vocabulary_size = max(train[feat].max(), test[feat].max()) + 1,
        embedding_dim   = 16,
    )
    for feat in sparse_features
]

# [改动2] dimension 从 1024 改成 64
dense_feature_columns = [
    DenseFeat(name="movie_llm_emb", dimension=PCA_DIM)
]

all_feature_columns = sparse_feature_columns + dense_feature_columns
feature_names       = get_feature_names(all_feature_columns)

# ── 5. 构造模型输入 ────────────────────────────────────────────────────────
train_input = {name: train[name].values for name in sparse_features}
test_input  = {name: test[name].values  for name in sparse_features}

# [改动3] 换成 PCA 压缩后的向量，不是原始1024维
train_input["movie_llm_emb"] = train_movie_emb_pca
test_input["movie_llm_emb"]  = test_movie_emb_pca

train_labels = train["label"].values
test_labels  = test["label"].values

# ── 6. 定义模型（和第4步完全一致）────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {device}")

model = DeepFM(
    linear_feature_columns = sparse_feature_columns,
    dnn_feature_columns    = all_feature_columns,
    dnn_hidden_units       = (256, 128),
    dnn_dropout            = 0.3,
    task                   = "binary",
    device                 = device,
)

# ── 7. 训练（和第4步完全一致）────────────────────────────────────────────
model.compile(
    optimizer = "adam",
    loss      = "binary_crossentropy",
    metrics   = ["auc"],
)

history = model.fit(
    x                = train_input,
    y                = train_labels,
    batch_size       = 4096,
    epochs           = 5,
    verbose          = 1,
    validation_split = 0.1,
)

# ── 8. 评估（和第4步完全一致）────────────────────────────────────────────
pred = model.predict(test_input, batch_size=4096)
auc  = roc_auc_score(test_labels, pred)
print(f"\n✅ PCA版双通道 DeepFM 全局 AUC: {auc:.4f}")

test_copy = test.copy()
test_copy["pred"] = pred

print("\n── 冷启动分层对比 ──")
for tier in ["very_cold", "cold", "warm"]:
    subset = test_copy[test_copy["coldstart_tier"] == tier]
    if len(subset) == 0:
        continue
    tier_auc = roc_auc_score(subset["label"], subset["pred"])
    print(f"  {tier:12s}: n={len(subset):6d}, AUC={tier_auc:.4f}")

# ── 9. 保存结果 ───────────────────────────────────────────────────────────
test_copy[["label", "pred", "coldstart_tier", "movie_id"]].to_csv(
    f"{OUT_DIR}/pca_pred.csv", index=False
)
print(f"\n预测结果已保存至 {OUT_DIR}/pca_pred.csv")