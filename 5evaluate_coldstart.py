"""
Step 5: 冷启动分层评估 + 四组对比可视化
四组：Baseline / PCA / Projection / Full 1024-dim
输出：
  fig1_auc_comparison.png          四组AUC分层对比柱状图
  fig2_coldstart_distribution.png  冷启动长尾分布
  fig3_auc_improvement.png         提升幅度对比（PCA vs Projection vs Full）
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import os

OUT_DIR = "./results"
FIG_DIR = "./figures"
os.makedirs(FIG_DIR, exist_ok=True)

# ── 1. 读取四组预测结果 ────────────────────────────────────────────────────
baseline = pd.read_csv(f"{OUT_DIR}/baseline_pred.csv")
pca      = pd.read_csv(f"{OUT_DIR}/pca_pred.csv")
dual     = pd.read_csv(f"{OUT_DIR}/dual_pred.csv")
proj     = pd.read_csv(f"{OUT_DIR}/proj_pred.csv")

# ── 2. 计算各分层 AUC ─────────────────────────────────────────────────────
tiers       = ["very_cold", "cold", "warm", "overall"]
tier_labels = ["Very Cold\n(<5)", "Cold\n(5~20)", "Warm\n(≥20)", "Overall"]

def get_aucs(df):
    result = {}
    for tier in ["very_cold", "cold", "warm"]:
        sub = df[df["coldstart_tier"] == tier]
        result[tier] = roc_auc_score(sub["label"], sub["pred"])
    result["overall"] = roc_auc_score(df["label"], df["pred"])
    return result

baseline_aucs = get_aucs(baseline)
pca_aucs      = get_aucs(pca)
proj_aucs     = get_aucs(proj)
dual_aucs     = get_aucs(dual)

print(f"{'Tier':<12} {'Baseline':>10} {'PCA64':>10} {'Proj64':>10} {'Full1024':>10} {'ΔPCA':>8} {'ΔProj':>8} {'ΔFull':>8}")
print("-" * 80)
for tier in tiers:
    b = baseline_aucs[tier]
    p = pca_aucs[tier]
    r = proj_aucs[tier]
    d = dual_aucs[tier]
    print(f"{tier:<12} {b:>10.4f} {p:>10.4f} {r:>10.4f} {d:>10.4f} {p-b:>+8.4f} {r-b:>+8.4f} {d-b:>+8.4f}")

# ── 3. 图1：四组 AUC 分层对比柱状图 ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 6))

x     = np.arange(len(tiers))
width = 0.2

bars1 = ax.bar(x - 1.5 * width, [baseline_aucs[t] for t in tiers],
               width, label="Baseline DeepFM", color="#5B8DB8", alpha=0.85)
bars2 = ax.bar(x - 0.5 * width, [pca_aucs[t] for t in tiers],
               width, label="+ LLM (PCA 64-dim)", color="#A8C256", alpha=0.85)
bars3 = ax.bar(x + 0.5 * width, [proj_aucs[t] for t in tiers],
               width, label="+ LLM (Proj 64-dim)", color="#F5C04A", alpha=0.85)
bars4 = ax.bar(x + 1.5 * width, [dual_aucs[t] for t in tiers],
               width, label="+ LLM (Full 1024-dim)", color="#E8724A", alpha=0.85)

for bars in [bars1, bars2, bars3, bars4]:
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.002,
                f"{bar.get_height():.4f}",
                ha="center", va="bottom", fontsize=6.5)

ax.set_xlabel("Cold-Start Tier", fontsize=12)
ax.set_ylabel("AUC", fontsize=12)
ax.set_title("AUC Comparison: Baseline vs PCA vs Projection vs Full LLM\n"
             "Trainable projection (Proj 64-dim) vs fixed PCA — which learns better CTR representations?",
             fontsize=12, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(tier_labels, fontsize=10)
ax.set_ylim(0.5, 0.84)
ax.legend(fontsize=9)
ax.yaxis.grid(True, alpha=0.3)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig1_auc_comparison.png", dpi=150, bbox_inches="tight")
print("\n✅ 图1已保存")
plt.close()

# ── 4. 图2：冷启动分布（长尾问题）────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

tier_counts = baseline.groupby("coldstart_tier").size().reindex(
    ["very_cold", "cold", "warm"]
)
colors = ["#E8724A", "#F5C04A", "#5B8DB8"]

bars = axes[0].bar(["Very Cold\n(<5)", "Cold\n(5-20)", "Warm\n(≥20)"],
                   tier_counts.values, color=colors, alpha=0.85)
for bar, val in zip(bars, tier_counts.values):
    axes[0].text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 500,
                 f"{val:,}", ha="center", fontsize=10)
axes[0].set_title("Sample Distribution by Cold-Start Tier", fontweight="bold")
axes[0].set_ylabel("Number of Interactions")
axes[0].yaxis.grid(True, alpha=0.3)
axes[0].set_axisbelow(True)

axes[1].pie(tier_counts.values,
            labels=[f"Very Cold\n({tier_counts['very_cold']:,})",
                    f"Cold\n({tier_counts['cold']:,})",
                    f"Warm\n({tier_counts['warm']:,})"],
            colors=colors, autopct="%1.1f%%",
            startangle=90, textprops={"fontsize": 10})
axes[1].set_title("Proportion of Cold-Start Interactions", fontweight="bold")

plt.suptitle("Long-Tail Problem in MovieLens-1M", fontsize=13,
             fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig2_coldstart_distribution.png", dpi=150, bbox_inches="tight")
print("✅ 图2已保存")
plt.close()

# ── 5. 图3：三种LLM方案 vs Baseline 提升幅度对比 ────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))

x     = np.arange(len(tiers))
width = 0.25

pca_imp  = [(pca_aucs[t]  - baseline_aucs[t]) * 100 for t in tiers]
proj_imp = [(proj_aucs[t] - baseline_aucs[t]) * 100 for t in tiers]
dual_imp = [(dual_aucs[t] - baseline_aucs[t]) * 100 for t in tiers]

bars1 = ax.bar(x - width, pca_imp,  width,
               label="+ LLM (PCA 64-dim)",    color="#A8C256", alpha=0.85)
bars2 = ax.bar(x,         proj_imp, width,
               label="+ LLM (Proj 64-dim)",   color="#F5C04A", alpha=0.85)
bars3 = ax.bar(x + width, dual_imp, width,
               label="+ LLM (Full 1024-dim)", color="#E8724A", alpha=0.85)

for bar, val in zip(bars1, pca_imp):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.05,
            f"+{val:.2f}%", ha="center", va="bottom", fontsize=8)
for bar, val in zip(bars2, proj_imp):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.05,
            f"+{val:.2f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")
for bar, val in zip(bars3, dual_imp):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.05,
            f"+{val:.2f}%", ha="center", va="bottom", fontsize=8)

ax.axhline(y=0, color="black", linewidth=0.8)
ax.set_ylabel("AUC Improvement over Baseline (%)", fontsize=11)
ax.set_title("PCA vs Projection vs Full-dim LLM: AUC Improvement over Baseline\n"
             "Trainable projection learns task-relevant semantics, unlike PCA's variance-based compression",
             fontsize=11, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(tier_labels, fontsize=10)
ax.legend(fontsize=9)
ax.yaxis.grid(True, alpha=0.3)
ax.set_axisbelow(True)
ax.set_ylim(0, max(dual_imp) * 1.35)

plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig3_auc_improvement.png", dpi=150, bbox_inches="tight")
print("✅ 图3已保存")
plt.close()

print(f"\n所有图表已保存至 {FIG_DIR}/")