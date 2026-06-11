"""
Step 4c: Projection Layer版双通道 DeepFM
与4b(PCA)的对比：不在输入侧用PCA降维，而是在DNN内部加一个可训练的projection层
核心创新：继承DeepFM，在DNN输入前加入 nn.Linear(1024→64)，让梯度决定哪些语义维度对CTR有用

方案演进：
  方案1 (4train_dualchannel) → 1024维直接concat，维度碾压稀疏特征
  方案2 (4b_train_pca)       → PCA降维，但PCA按方差压缩，不关心CTR任务目标
  方案3 (本脚本)              → Projection层，可训练，按CTR任务目标压缩
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from deepctr_torch.models import DeepFM
from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names, combined_dnn_input
from sklearn.metrics import roc_auc_score
import os


# ── 0. 自定义DeepFM子类：DNN输入侧加Projection ─────────────────────────────

class ProjDeepFM(DeepFM):
    """DeepFM with trainable projection layer for LLM dense features.

    在父类 DeepFM 的基础上，对 DNN 输入中的 LLM 稠密特征先过一层
    nn.Linear(llm_dim, proj_dim)，再和其他稀疏embedding拼接进DNN。

    和PCA的区别：
      - PCA 按"保留最大方差"方向压缩，不关心CTR任务
      - Projection 层通过反向传播学习"对CTR预测有用的方向"
    """

    def __init__(self, *args, llm_dim=1024, proj_dim=64, **kwargs):
        super().__init__(*args, **kwargs)
        self.llm_projection = nn.Linear(llm_dim, proj_dim, bias=False)
        self.llm_projection = self.llm_projection.to(self.device)

        # 父类 DNN 按 DenseFeat(dim=1024) 初始化了 Linear(1120, 256)
        # 但 projection 后实际输入是 96 + 64 = 160 维，需要重建 DNN
        dnn_hidden_units = [layer.out_features for layer in self.dnn.linears]
        orig_in_dim = self.dnn.linears[0].in_features
        corrected_in_dim = orig_in_dim - llm_dim + proj_dim

        from deepctr_torch.layers import DNN
        self.dnn = DNN(
            corrected_in_dim, dnn_hidden_units,
            activation='relu',
            dropout_rate=self.dnn.dropout_rate,
            use_bn=self.dnn.use_bn,
            device=self.device,
        )
        self.dnn_linear = nn.Linear(
            dnn_hidden_units[-1], 1, bias=False).to(self.device)

    def forward(self, X):
        # Step 1: 提取稀疏embedding和稠密特征值（同父类）
        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(
            X, self.dnn_feature_columns, self.embedding_dict)

        # ① Linear 部分（同父类）
        logit = self.linear_model(X)

        # ② FM 部分（同父类）
        if self.use_fm and len(sparse_embedding_list) > 0:
            fm_input = torch.cat(sparse_embedding_list, dim=1)
            logit += self.fm(fm_input)

        # ③ DNN 部分 —— 唯一改动：LLM向量先过projection再拼接
        if self.use_dnn:
            projected_dense = []
            for dv in dense_value_list:
                if dv.shape[-1] == self.llm_projection.in_features:
                    dv = self.llm_projection(dv)          # 1024 → 64
                projected_dense.append(dv)
            dnn_input = combined_dnn_input(sparse_embedding_list, projected_dense)
            dnn_output = self.dnn(dnn_input)
            dnn_logit = self.dnn_linear(dnn_output)
            logit += dnn_logit

        y_pred = self.out(logit)
        return y_pred


# ── 1. 路径配置 ────────────────────────────────────────────────────────────
DATA_DIR = "./data/processed"
EMB_DIR  = "./data/embeddings"
OUT_DIR  = "./results"
os.makedirs(OUT_DIR, exist_ok=True)

# ── 2. 读数据 ──────────────────────────────────────────────────────────────
train = pd.read_csv(f"{DATA_DIR}/train.csv")
test  = pd.read_csv(f"{DATA_DIR}/test.csv")
print(f"train: {train.shape}, test: {test.shape}")

# ── 3. 加载 LLM 向量矩阵 ──────────────────────────────────────────────────
emb_matrix = np.load(f"{EMB_DIR}/movie_embeddings.npy")
print(f"LLM向量矩阵: {emb_matrix.shape}")

train_movie_emb = emb_matrix[train["movie_id"].values]   # (800167, 1024)
test_movie_emb  = emb_matrix[test["movie_id"].values]    # (200042, 1024)

# ── 4. 定义特征列（和4train_dualchannel完全一致）──────────────────────────
# 注意：DenseFeat仍然是1024维，降维由模型内部的projection层完成
# 这和PCA的区别：PCA在输入侧降维 → 不可训练；Projection在模型内降维 → 可训练

sparse_features = ["user_id", "movie_id", "gender", "age", "occupation", "zip"]
DENSE_DIM = 1024
PROJ_DIM  = 64

sparse_feature_columns = [
    SparseFeat(
        name            = feat,
        vocabulary_size = max(train[feat].max(), test[feat].max()) + 1,
        embedding_dim   = 16,
    )
    for feat in sparse_features
]

dense_feature_columns = [
    DenseFeat(name="movie_llm_emb", dimension=DENSE_DIM)
]

all_feature_columns = sparse_feature_columns + dense_feature_columns
feature_names       = get_feature_names(all_feature_columns)
print(f"特征列表: {feature_names}")

# ── 5. 构造模型输入 ────────────────────────────────────────────────────────
train_input = {name: train[name].values for name in sparse_features}
test_input  = {name: test[name].values  for name in sparse_features}

# 传的是原始1024维，降维交给模型内部的projection层
train_input["movie_llm_emb"] = train_movie_emb
test_input["movie_llm_emb"]  = test_movie_emb

train_labels = train["label"].values
test_labels  = test["label"].values

# ── 6. 定义 Projection 版双通道模型 ──────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {device}")

model = ProjDeepFM(
    linear_feature_columns = sparse_feature_columns,
    dnn_feature_columns    = all_feature_columns,
    dnn_hidden_units       = (256, 128),
    dnn_dropout            = 0.3,
    task                   = "binary",
    device                 = device,
    llm_dim                = DENSE_DIM,       # 传给子类的projection层
    proj_dim               = PROJ_DIM,
)

# ── 7. 训练 ───────────────────────────────────────────────────────────────
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

# ── 8. 评估 ───────────────────────────────────────────────────────────────
pred = model.predict(test_input, batch_size=4096)
auc  = roc_auc_score(test_labels, pred)
print(f"\n✅ Projection版双通道 DeepFM 全局 AUC: {auc:.4f}")

test_copy = test.copy()
test_copy["pred"] = pred

print("\n── 冷启动分层评估 ──")
for tier in ["very_cold", "cold", "warm"]:
    subset = test_copy[test_copy["coldstart_tier"] == tier]
    if len(subset) == 0:
        continue
    tier_auc = roc_auc_score(subset["label"], subset["pred"])
    print(f"  {tier:12s}: n={len(subset):6d}, AUC={tier_auc:.4f}")

# ── 9. 保存结果 ───────────────────────────────────────────────────────────
test_copy[["label", "pred", "coldstart_tier", "movie_id"]].to_csv(
    f"{OUT_DIR}/proj_pred.csv", index=False
)
print(f"\n预测结果已保存至 {OUT_DIR}/proj_pred.csv")
