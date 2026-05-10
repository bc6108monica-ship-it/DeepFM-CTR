"""
Step 3: 训练 Baseline DeepFM
功能：只用 Sparse ID 特征，不加 LLM 向量，跑出对照组 AUC
这个数字是后面所有对比的基准
"""

import pandas as pd
import numpy as np
import torch
from deepctr_torch.models import DeepFM
from deepctr_torch.inputs import SparseFeat, get_feature_names
from sklearn.metrics import roc_auc_score
import joblib, os

# ── 0. 路径配置 ────────────────────────────────────────────────────────────
DATA_DIR = "./data/processed"
OUT_DIR  = "./results"
os.makedirs(OUT_DIR, exist_ok=True)

# ── 1. 读数据 ──────────────────────────────────────────────────────────────
train = pd.read_csv(f"{DATA_DIR}/train.csv")
test  = pd.read_csv(f"{DATA_DIR}/test.csv")

print(f"train: {train.shape}, test: {test.shape}")

# ── 2. 定义特征列 ──────────────────────────────────────────────────────────
# SparseFeat 三个关键参数：这个是要告诉Deepfm的
# 每个类别特征的名字、这个特征有多少个不同的值（决定 embedding 表的行数）、每个值映射成多少维的向量
#   name       : 列名
#   vocabulary_size : 这个特征有多少个不同的值（决定 embedding 表的行数）
#   embedding_dim   : 每个值映射成多少维的向量

sparse_features = ["user_id", "movie_id", "gender", "age", "occupation", "zip"]

fixlen_feature_columns = [
    SparseFeat(
        name            = feat,
        vocabulary_size = max(train[feat].max(), test[feat].max()) + 1,
        embedding_dim   = 16,   # 先用16维，够用且训练快
    )
    for feat in sparse_features
]

# get_feature_names 返回模型实际需要的输入列名列表
feature_names = get_feature_names(fixlen_feature_columns)
print(f"特征列表: {feature_names}")

# ── 3. 构造模型输入 ────────────────────────────────────────────────────────
#转化成字典格式，DeepFM 的输入格式是字典：{特征名: numpy_array}
# DeepCTR 的输入格式是字典：KEY:特征名，VALUE: numpy_array  
# {特征名: numpy_array}
train_input = {name: train[name].values for name in feature_names}
test_input  = {name: test[name].values  for name in feature_names}

train_labels = train["label"].values
test_labels  = test["label"].values

# ── 4. 定义模型 ────────────────────────────────────────────────────────────
# DeepFM 参数说明：
#   linear_feature_columns : 进入 FM 线性部分的特征
#   dnn_feature_columns    : 进入 DNN 部分的特征
#   两个参数传一样的，所有特征都同时走 FM 和 DNN
#   dnn_hidden_units       : DNN 每层的神经元数，(256, 128) 表示两层
#   task                   : 'binary' 二分类，输出 sigmoid

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {device}")
print(torch.cuda.get_device_name(0)) 

model = DeepFM(
    linear_feature_columns = fixlen_feature_columns,
    dnn_feature_columns    = fixlen_feature_columns,
    dnn_hidden_units       = (256, 128),
    dnn_dropout            = 0.3,#加点 dropout 看看能不能提升泛化，过拟合了的话可以调大一点
    task                   = "binary",# 2分类问题，输出 sigmoid
    device                 = device,
)

# ── 5. 编译和训练 ──────────────────────────────────────────────────────────
model.compile(
    optimizer = "adam",
    loss      = "binary_crossentropy",
    metrics   = ["auc"],
)

history = model.fit(
    x          = train_input,
    y          = train_labels,
    batch_size = 4096,
    epochs     = 5,
    verbose    = 1,
    validation_split = 0.1,   # 从训练集里拿10%做验证，监控过拟合
)

# ── 6. 在测试集上评估 ──────────────────────────────────────────────────────
pred = model.predict(test_input, batch_size=4096)
auc  = roc_auc_score(test_labels, pred)
print(f"\n✅ Baseline DeepFM 全局 AUC: {auc:.4f}")

# ── 7. 冷启动分层评估 ──────────────────────────────────────────────────────
# 把预测结果加回 test dataframe
test = test.copy()
test["pred"] = pred

for tier in ["very_cold", "cold", "warm"]:
    subset = test[test["coldstart_tier"] == tier]
    if len(subset) == 0:
        continue
    tier_auc = roc_auc_score(subset["label"], subset["pred"])
    print(f"  {tier:12s}: n={len(subset):6d}, AUC={tier_auc:.4f}")

# ── 8. 保存结果供后面对比用 ───────────────────────────────────────────────
results = {
    "model"    : "baseline_deepfm",
    "global_auc": auc,
}
# 把预测结果存下来，第五步统一对比用
test[["label", "pred", "coldstart_tier", "movie_id"]].to_csv(
    f"{OUT_DIR}/baseline_pred.csv", index=False
)
print(f"\n预测结果已保存至 {OUT_DIR}/baseline_pred.csv")