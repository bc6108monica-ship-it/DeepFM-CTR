"""
Step 5: 冷启动分层评估 + 可视化
输出三张图：
  1. AUC 分层对比柱状图（核心结论图）
  2. 冷启动分布图（展示长尾问题）
  3. 提升幅度图（突出冷启动的改善效果）
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import roc_auc_score
import os

# 解决中文字体问题
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

OUT_DIR    = "./results"
FIG_DIR    = "./figures"
os.makedirs(FIG_DIR, exist_ok=True)

# ── 1. 读取两个模型的预测结果 ──────────────────────────────────────────────
baseline = pd.read_csv(f"{OUT_DIR}/baseline_pred.csv")
dual     = pd.read_csv(f"{OUT_DIR}/dual_pred.csv")

print(f"baseline: {baseline.shape}")
print(f"dual:     {dual.shape}")

# ── 2. 计算各分层 AUC ─────────────────────────────────────────────────────
tiers       = ["very_cold", "cold", "warm", "overall"]
tier_labels = ["Very Cold\n(<5 interactions)", "Cold\n(5-20)", "Warm\n(≥20)", "Overall"]

def get_auc_by_tier(df):
    results = {}
    for tier in ["very_cold", "cold", "warm"]:
        subset = df[df["coldstart_tier"] == tier]
        results[tier] = roc_auc_score(subset["label"], subset["pred"])
    results["overall"] = roc_auc_score(df["label"], df["pred"])
    return results

baseline_aucs = get_auc_by_tier(baseline)
dual_aucs     = get_auc_by_tier(dual)

print("\n── AUC 对比 ──")
print(f"{'Tier':<12} {'Baseline':>10} {'Dual':>10} {'Δ AUC':>10}")
print("-" * 45)
for tier, label in zip(tiers, tier_labels):
    b = baseline_aucs[tier]
    d = dual_aucs[tier]
    print(f"{tier:<12} {b:>10.4f} {d:>10.4f} {d-b:>+10.4f}")

# ── 3. 图1：AUC 分层对比柱状图 ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))

x      = np.arange(len(tiers))
width  = 0.35

bars1 = ax.bar(x - width/2,
               [baseline_aucs[t] for t in tiers],
               width, label="Baseline DeepFM", color="#5B8DB8", alpha=0.85)
bars2 = ax.bar(x + width/2,
               [dual_aucs[t] for t in tiers],
               width, label="Dual-Channel DeepFM (+ LLM)", color="#E8724A", alpha=0.85)

# 在柱子上标注数值
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
            f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=9)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
            f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=9)

ax.set_xlabel("Cold-Start Tier", fontsize=12)
ax.set_ylabel("AUC", fontsize=12)
ax.set_title("AUC Comparison by Cold-Start Tier\nBaseline DeepFM vs Dual-Channel DeepFM with LLM Embeddings",
             fontsize=13, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(tier_labels, fontsize=10)
ax.set_ylim(0.5, 0.82)
ax.legend(fontsize=11)
ax.yaxis.grid(True, alpha=0.3)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig1_auc_comparison.png", dpi=150, bbox_inches="tight")
print("\n✅ 图1已保存：fig1_auc_comparison.png")
plt.close()

# ── 4. 图2：冷启动分布图（长尾问题可视化）────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 左图：各分层样本数量
tier_counts = baseline.groupby("coldstart_tier").size().reindex(
    ["very_cold", "cold", "warm"]
)
colors = ["#E8724A", "#F5C04A", "#5B8DB8"]
bars = axes[0].bar(["Very Cold\n(<5)", "Cold\n(5-20)", "Warm\n(≥20)"],
                   tier_counts.values, color=colors, alpha=0.85)
for bar, val in zip(bars, tier_counts.values):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
                 f"{val:,}", ha="center", fontsize=10)
axes[0].set_title("Sample Distribution by Cold-Start Tier", fontweight="bold")
axes[0].set_ylabel("Number of Interactions")
axes[0].yaxis.grid(True, alpha=0.3)
axes[0].set_axisbelow(True)

# 右图：各分层占比饼图
axes[1].pie(tier_counts.values,
            labels=[f"Very Cold\n({tier_counts['very_cold']:,})",
                    f"Cold\n({tier_counts['cold']:,})",
                    f"Warm\n({tier_counts['warm']:,})"],
            colors=colors, autopct="%1.1f%%",
            startangle=90, textprops={"fontsize": 10})
axes[1].set_title("Proportion of Cold-Start Interactions", fontweight="bold")

plt.suptitle("Long-Tail Problem in MovieLens-1M", fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig2_coldstart_distribution.png", dpi=150, bbox_inches="tight")
print("✅ 图2已保存：fig2_coldstart_distribution.png")
plt.close()

# ── 5. 图3：AUC 提升幅度图 ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))

improvements = [(dual_aucs[t] - baseline_aucs[t]) * 100 for t in tiers]
colors_imp   = ["#E8724A" if v > 0.5 else "#F5C04A" if v > 0.1 else "#5B8DB8"
                for v in improvements]

bars = ax.bar(tier_labels, improvements, color=colors_imp, alpha=0.85, width=0.5)

for bar, val in zip(bars, improvements):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.05,
            f"+{val:.2f}%", ha="center", va="bottom",
            fontsize=11, fontweight="bold")

ax.axhline(y=0, color="black", linewidth=0.8)
ax.set_ylabel("AUC Improvement (%)", fontsize=12)
ax.set_title("AUC Improvement of Dual-Channel over Baseline\n(LLM Embeddings help most on cold-start items)",
             fontsize=12, fontweight="bold")
ax.yaxis.grid(True, alpha=0.3)
ax.set_axisbelow(True)
ax.set_ylim(0, max(improvements) * 1.3)

plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig3_auc_improvement.png", dpi=150, bbox_inches="tight")
print("✅ 图3已保存：fig3_auc_improvement.png")
plt.close()

print(f"\n所有图表已保存至 {FIG_DIR}/")
print("可以直接放进 README 和简历附件里")