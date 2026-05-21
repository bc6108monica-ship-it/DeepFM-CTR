# DeepFM-CTR

Dual-channel DeepFM with LLM semantic embeddings for cold-start CTR prediction on MovieLens-1M.

## 项目总览（三层架构）

```mermaid
graph TD
    %% ── 样式定义 ──
    classDef layer fill:#f0f0f0,stroke:#333,stroke-width:2px,font-weight:bold
    classDef func fill:#E3F2FD,stroke:#1565C0,stroke-width:1px
    classDef sys  fill:#F3E5F5,stroke:#7B1FA2,stroke-width:1px
    classDef eng  fill:#FFF3E0,stroke:#E65100,stroke-width:1px
    classDef result fill:#C8E6C9,stroke:#2E7D32,stroke-width:1px

    %% ══════════ L1: 功能层 ══════════
    subgraph L1["第1层 · 功能层：跑通 Pipeline，拿到结果"]
        direction LR
        A1["数据预处理<br/>1data_process.py"] --> A2["LLM 向量生成<br/>2generate_embeddings.py"]
        A2 --> A3["三组模型训练<br/>3/4/4b"]
        A3 --> A4["评估 & 可视化<br/>5evaluate_coldstart.py"]
    end

    %% ══════════ L2: 系统层 ══════════
    subgraph L2["第2层 · 系统层：模块设计 & 交互"]
        direction TB
        B1["模块A：数据预处理<br/>• 按时间序切分 8:2<br/>• LabelEncoder 编码<br/>• 冷启动分层标记"] --> B2["模块B：LLM 语义向量<br/>• 调智谱 Embedding-3 API<br/>• 批量 64 条/次<br/>• 存为 (N×1024) 矩阵"]
        B2 --> B3["模块C：模型训练"]
        B3 --> B4["模块D：评估对比<br/>• 全局 AUC<br/>• 分层 AUC"]
    end

    B3 --> B3a["Baseline DeepFM<br/>仅 Sparse ID 特征"]
    B3 --> B3b["Dual-Channel<br/>Sparse + 1024d LLM"]
    B3 --> B3c["PCA 消融<br/>Sparse + 64d PCA"]

    %% ══════════ L3: 工程层 ══════════
    subgraph L3["第3层 · 工程层：上线考量（实验阶段，标注待完善项）"]
        direction LR
        C1["性能<br/>• 80万样本 5 epoch ~分钟级<br/>• embedding 矩阵仅 15MB<br/>• ⚠️ 大数据量需换 Spark"]
        C2["成本<br/>• LLM API 不到 1 元<br/>• GPU 训练近零成本<br/>• ⚠️ 10x 数据量 embedding 表膨胀"]
        C3["稳定性<br/>• ⚠️ 未固定随机种子<br/>• ⚠️ 缺 LabelEncoder 持久化<br/>• ⚠️ 缺 unseen 值兜底"]
        C4["可观测 & 部署<br/>• ⚠️ 缺 TensorBoard/MLflow<br/>• ⚠️ 缺模型 checkpoint<br/>• 上线: ONNX + Redis 缓存"]
    end

    %% ── 层级间的关系 ──
    L1 -.->|支撑| L2
    L2 -.->|指导| L3

    %% 应用样式
    class L1,L2,L3 layer
```

## 核心思路

将电影标题和类型通过 LLM 转为语义 embedding（1024维），作为 **冻结的稠密特征** 拼接到 Dual-Channel DeepFM 的第二个通道中，缓解冷启动用户/物品的 AUC 衰退问题。

## 实验结果

| Tier | Baseline DeepFM | Dual-Channel (1024d) | PCA消融 (64d) | Δ 1024-PCA |
|------|:-:|:-:|:-:|:-:|
| Very Cold (<5) | 0.5680 | **0.6093** | 0.6018 | +0.0075 |
| Cold (5-20) | 0.6558 | **0.6692** | 0.6673 | +0.0019 |
| Warm (≥20) | 0.7446 | 0.7447 | 0.7445 | +0.0003 |
| **Overall** | 0.7442 | **0.7445** | 0.7442 | +0.0003 |

**关键结论：**
- **LLM embedding 有效**：1024维双通道在冷启动层显著提升（Very Cold +4.13%），热启动基本持平
- **PCA 降维有损**：1024→64维压缩后冷启动增益折半以上（Very Cold 从 +4.13% 降至 +3.38%），说明语义信息的完整性对冷启动很重要

## 可视化图表

| 图 | 说明 |
|----|------|
| ![fig1](figures/fig1_auc_comparison.png) | AUC 分层对比柱状图 |
| ![fig2](figures/fig2_coldstart_distribution.png) | 冷启动分布图（长尾问题可视化） |
| ![fig3](figures/fig3_auc_improvement.png) | 提升幅度图 |

## 项目结构

```
├── 1data_process.py             # 数据预处理
├── 2generate_embeddings.py      # 调用 LLM 生成语义 embedding
├── 3train_baseline.py           # Baseline DeepFM 训练
├── 4train_dualchannel.py        # Dual-Channel DeepFM 训练
├── 5evaluate_coldstart.py       # PCA消融实验（对比验证）
├── results/
│   ├── baseline_pred.csv
│   ├── dual_pred.csv
│   └── pca_pred.csv
├── figures/
│   ├── fig1_auc_comparison.png
│   ├── fig2_coldstart_distribution.png
│   └── fig3_auc_improvement.png
└── README.md
```
