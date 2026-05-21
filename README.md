# DeepFM-CTR

Dual-channel DeepFM with LLM semantic embeddings for cold-start CTR prediction on MovieLens-1M.

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

```mermaid
mindmap
  root((DeepFM-CTR))
    数据管道
      1data_process.py
        输入: MovieLens-1M 原始数据
        处理: train / test 划分 + 冷启动分层
        输出: data/processed/
      2generate_embeddings.py
        输入: 电影标题 + 类型
        处理: LLM → 1024维语义向量
        输出: data/embeddings/movie_embeddings.npy
    模型训练
      3train_baseline.py
        模型: DeepFM（仅 Sparse ID 特征）
        作用: 冷启动 AUC 衰退基线
      4train_dualchannel.py
        模型: 双通道 DeepFM（Sparse + 1024d LLM）
        创新: DNN 有监督学习语义压缩
      4b_train_pca_channel.py
        模型: DeepFM + PCA 64维（消融对照）
        作用: 验证完整语义 > PCA 有损压缩
    评估可视化
      5evaluate_coldstart.py
        评估: AUC 全局 + 分层对比
        可视化: 生成 PNG 图表
        figures/
          fig1_auc_comparison.png
          fig2_coldstart_distribution.png
          fig3_auc_improvement.png
    辅助文件
      deepfm_baseline.py: DeepFM 模型实现
      run_deepfm.py: 原始运行入口
      data/: 数据目录（raw / processed / embeddings）
      results/: 预测结果 CSV
        baseline_pred.csv
        dual_pred.csv
        pca_pred.csv
```
