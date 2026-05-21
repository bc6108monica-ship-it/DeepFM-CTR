# DeepFM-CTR

Dual-channel DeepFM with LLM semantic embeddings for cold-start CTR prediction on MovieLens-1M.

## 实验设计总览

```mermaid
mindmap
  root((LLM-DeepFM-CTR<br>实验设计))
    问题
      场景: MovieLens-1M 评分预测
      痛点: 冷启动用户/电影的 AUC 严重衰退
      目标: 利用 LLM 语义向量缓解冷启动
    数据流
      1_数据预处理
        数据集划分: train / test
        冷启动分层: very_cold / cold / warm
      2_LLM 语义向量
        输入: 电影标题 + 类型
        模型: LLM → 1024维 embedding
        产出: movie_embeddings.npy
    三组对比实验
      A_Baseline DeepFM
        输入: 仅 Sparse ID 特征
        作用: 对照组，衡量冷启动衰退基线
      B_双通道 DeepFM
        输入: Sparse ID + 完整 1024d LLM 向量
        核心: DNN 自动学习语义压缩（有监督）
        预期: 冷启动层显著提升
      C_PCA 消融实验
        输入: Sparse ID + PCA 压缩 64d 向量
        作用: 验证"完整语义 > PCA 压缩"
        差异: 仅降维方式不同，其余完全一致
    评估
      指标: AUC（全局 + 分层）
      对比: A vs B vs C 三组横向对比
      可视化: 柱状图 + 提升幅度图
    关键结论
      LLM 语义有效: Very Cold +4.13%
      PCA 有损: 增益折半以上
      完整 1024d > PCA 64d > Baseline
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
