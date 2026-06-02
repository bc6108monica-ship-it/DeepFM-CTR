"""
Step 5b: Bootstrap 假设检验 — AUC 差异的统计显著性
对三组模型 × 四个层级 做 paired bootstrap 重采样，
输出 95% CI、p-value、以及分布直方图。
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import os

OUT_DIR = "./results"
FIG_DIR = "./figures"
os.makedirs(FIG_DIR, exist_ok=True)

N_BOOTSTRAP = 2000
ALPHA = 0.05
SEED = 42
np.random.seed(SEED)

# ── 1. 读取三组预测 ──────────────────────────────────────────────────────
baseline = pd.read_csv(f"{OUT_DIR}/baseline_pred.csv")
pca      = pd.read_csv(f"{OUT_DIR}/pca_pred.csv")
dual     = pd.read_csv(f"{OUT_DIR}/dual_pred.csv")

datasets = {"Baseline": baseline, "PCA-64": pca, "Dual-1024": dual}

# ── 2. Bootstrap 函数 — 配对重采样（三个模型用同一组 bootstrap 索引）────
def bootstrap_auc_diff(df_a, df_b, tier, n_boot=N_BOOTSTRAP):
    """
    配对 bootstrap 计算模型 A vs 模型 B 在指定 tier 的 AUC 差值分布。
    返回: (mean_diff, ci_lower, ci_upper, p_two_sided, diffs_array)

    tier = "overall" 则用全量数据，否则按 coldstart_tier 筛选。
    """
    if tier == "overall":
        sub_a, sub_b = df_a, df_b
    else:
        sub_a = df_a[df_a["coldstart_tier"] == tier]
        sub_b = df_b[df_b["coldstart_tier"] == tier]

    y_a = sub_a["label"].values
    p_a = sub_a["pred"].values
    p_b = sub_b["pred"].values

    n = len(y_a)
    diffs = np.empty(n_boot)

    for i in range(n_boot):
        idx = np.random.choice(n, size=n, replace=True)
        auc_a = roc_auc_score(y_a[idx], p_a[idx])
        auc_b = roc_auc_score(y_a[idx], p_b[idx])   # 同 idx，保证配对
        diffs[i] = auc_b - auc_a

    mean_diff = diffs.mean()
    ci_lower  = np.percentile(diffs, ALPHA / 2 * 100)
    ci_upper  = np.percentile(diffs, (1 - ALPHA / 2) * 100)

    # 双尾 p 值: H0 是 diff=0，统计量为 mean(diff) 偏离 0 的程度
    p_two_sided = min(
        (diffs <= 0).mean(),     # left tail
        (diffs >= 0).mean()      # right tail
    ) * 2
    p_two_sided = min(p_two_sided, 1.0)

    return mean_diff, ci_lower, ci_upper, p_two_sided, diffs

# ── 3. 运行全部检验 ──────────────────────────────────────────────────────
tiers = ["very_cold", "cold", "warm", "overall"]
comparisons = [
    ("Dual-1024", "Baseline"),
    ("PCA-64",    "Baseline"),
    ("Dual-1024", "PCA-64"),
]

print("=" * 90)
print("Bootstrap 假设检验 — AUC 差异 (paired bootstrap, n={}, α={})".format(N_BOOTSTRAP, ALPHA))
print("=" * 90)

results = {}  # (comp, tier) -> dict

for comp_name_a, comp_name_b in comparisons:
    df_a = datasets[comp_name_a]
    df_b = datasets[comp_name_b]
    label = f"{comp_name_a} vs {comp_name_b}"
    print(f"\n{'─' * 80}")
    print(f"  {label}")
    print(f"{'Tier':<14} {'ΔAUC':>8} {'95% CI':>22} {'p-value':>9}  {'Significant(α=0.05)':>20}")
    print(f"{'─' * 80}")

    for tier in tiers:
        mean_diff, ci_lo, ci_hi, p_val, diffs = bootstrap_auc_diff(df_a, df_b, tier)
        results[(label, tier)] = {
            "mean_diff": mean_diff,
            "ci_lower": ci_lo,
            "ci_upper": ci_hi,
            "p_value": p_val,
            "diffs": diffs,
        }
        sig = "✅ YES" if p_val < ALPHA else "✗ no"
        print(f"  {tier:<12}  {mean_diff:+.4f}   [{ci_lo:+.4f}, {ci_hi:+.4f}]   {p_val:.4f}     {sig:>20}")

# ── 4. 汇总表（适合答辩/简历引用）────────────────────────────────────────
print(f"\n\n{'=' * 90}")
print("汇总 — 关键结论（仅 very_cold + cold 层）")
print(f"{'=' * 90}")
for comp_a, comp_b in comparisons:
    comp_label = f"{comp_a} vs {comp_b}"
    for tier in ["very_cold", "cold"]:
        r = results[(comp_label, tier)]
        print(f"  {comp_label:28s} | {tier:12s} | ΔAUC = {r['mean_diff']:+.4f} "
              f"95% CI [{r['ci_lower']:+.4f}, {r['ci_upper']:+.4f}] "
              f"p = {r['p_value']:.4f}")

# ── 5. 图4：Bootstrap 分布直方图 — Dual vs Baseline in Very Cold ─────────
key_comparisons_for_plot = [
    ("Dual-1024 vs Baseline", "very_cold"),
    ("PCA-64 vs Baseline",    "very_cold"),
    ("Dual-1024 vs Baseline", "cold"),
    ("Dual-1024 vs Baseline", "warm"),
]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for ax, (comp_label, tier) in zip(axes, key_comparisons_for_plot):
    r = results[(comp_label, tier)]
    diffs = r["diffs"]

    ax.hist(diffs, bins=50, color="#5B8DB8" if "Baseline" in comp_label else "#E8724A",
            alpha=0.75, edgecolor="white", linewidth=0.3)
    ax.axvline(x=0, color="black", linewidth=1.2, linestyle="--", label="H₀: ΔAUC = 0")
    ax.axvline(x=r["mean_diff"], color="red", linewidth=1.5,
               label=f"Mean Δ = {r['mean_diff']:+.4f}")
    ax.axvline(x=r["ci_lower"], color="gray", linewidth=0.8, linestyle=":")
    ax.axvline(x=r["ci_upper"], color="gray", linewidth=0.8, linestyle=":")

    tier_name = {"very_cold": "Very Cold (<5)", "cold": "Cold (5-20)", "warm": "Warm (≥20)"}[tier]
    ax.set_title(f"{comp_label}\n{tier_name} — p = {r['p_value']:.4f}",
                 fontsize=10, fontweight="bold")
    ax.set_xlabel("ΔAUC (Bootstrap)")
    ax.set_ylabel("Frequency")
    ax.legend(fontsize=8, loc="upper right")
    ax.yaxis.grid(True, alpha=0.2)
    ax.set_axisbelow(True)

plt.suptitle("Bootstrap Distribution of AUC Differences (n=2000)\n"
             "Solid red = observed mean; Dotted gray = 95% CI; Dashed black = null hypothesis",
             fontsize=12, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig4_bootstrap_distribution.png", dpi=150, bbox_inches="tight")
print(f"\n✅ 图4已保存至 {FIG_DIR}/fig4_bootstrap_distribution.png")
plt.close()

# ── 6. 图5：所有比较的森林图（Forest Plot） ──────────────────────────────
all_comps = [
    ("Dual vs Base", "very_cold"),
    ("PCA vs Base",  "very_cold"),
    ("Dual vs PCA",  "very_cold"),
    ("Dual vs Base", "cold"),
    ("PCA vs Base",  "cold"),
    ("Dual vs PCA",  "cold"),
    ("Dual vs Base", "warm"),
    ("PCA vs Base",  "warm"),
    ("Dual vs PCA",  "warm"),
]

comp_label_map = {
    ("Dual-1024 vs Baseline", "very_cold"): ("Dual vs Base", "very_cold"),
    ("PCA-64 vs Baseline",    "very_cold"): ("PCA vs Base",  "very_cold"),
    ("Dual-1024 vs PCA-64",   "very_cold"): ("Dual vs PCA",  "very_cold"),
    ("Dual-1024 vs Baseline", "cold"):      ("Dual vs Base", "cold"),
    ("PCA-64 vs Baseline",    "cold"):      ("PCA vs Base",  "cold"),
    ("Dual-1024 vs PCA-64",   "cold"):      ("Dual vs PCA",  "cold"),
    ("Dual-1024 vs Baseline", "warm"):      ("Dual vs Base", "warm"),
    ("PCA-64 vs Baseline",    "warm"):      ("PCA vs Base",  "warm"),
    ("Dual-1024 vs PCA-64",   "warm"):      ("Dual vs PCA",  "warm"),
}

fig, ax = plt.subplots(figsize=(12, 6))

y_positions = []
y_labels = []
means = []
ci_lows = []
ci_highs = []
colors = []
sig_markers = []

for i, (short_name, tier) in enumerate(all_comps):
    # reverse lookup
    full_key = None
    for (cl, t), (sn, tt) in comp_label_map.items():
        if sn == short_name and tt == tier:
            full_key = (cl, t)
            break

    r = results[full_key]
    y_positions.append(i)
    tier_display = {"very_cold": "Very Cold", "cold": "Cold", "warm": "Warm"}[tier]
    y_labels.append(f"{short_name}\n[{tier_display}]")
    means.append(r["mean_diff"])
    ci_lows.append(r["ci_lower"])
    ci_highs.append(r["ci_upper"])

    if r["p_value"] < 0.01:
        colors.append("#E8724A")  # highly significant
    elif r["p_value"] < 0.05:
        colors.append("#F5C04A")  # significant
    else:
        colors.append("#5B8DB8")  # not significant

    sig_str = "***" if r["p_value"] < 0.01 else ("**" if r["p_value"] < 0.05 else "ns")
    sig_markers.append(sig_str)

ax.axvline(x=0, color="black", linewidth=1.0, linestyle="--", alpha=0.6)

for i in range(len(y_positions)):
    ax.errorbar(means[i], y_positions[i],
                xerr=[[means[i] - ci_lows[i]], [ci_highs[i] - means[i]]],
                fmt="o", color=colors[i], capsize=4, markersize=8,
                markeredgewidth=1.5, markeredgecolor="white")
    ax.text(means[i] + 0.002, y_positions[i], sig_markers[i],
            va="center", fontsize=12, fontweight="bold", color=colors[i])

ax.set_yticks(y_positions)
ax.set_yticklabels(y_labels, fontsize=9)
ax.set_xlabel("AUC Difference (Bootstrap, n=2000)", fontsize=11)
ax.set_title("Forest Plot: AUC Differences with 95% Bootstrap CI\n"
             "*** p<0.01  ** p<0.05  ns = not significant",
             fontsize=12, fontweight="bold")
ax.set_ylim(-1, len(y_positions))
ax.xaxis.grid(True, alpha=0.3)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig5_forest_plot.png", dpi=150, bbox_inches="tight")
print(f"✅ 图5已保存至 {FIG_DIR}/fig5_forest_plot.png")
plt.close()

print(f"\n所有 Bootstrap 检验完成 ✅")
