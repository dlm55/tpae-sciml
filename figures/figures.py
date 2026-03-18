"""
figures.py
==========
Generates all publication-quality figures for the project.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

os.makedirs("figures", exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.titlesize":    12,
    "axes.labelsize":    11,
    "legend.fontsize":   9,
    "figure.dpi":        150,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

COLORS = {
    "physics":  "#2166ac",   # blue
    "baseline": "#d73027",   # red
    "guided":   "#1a9641",   # green
    "data":     "#333333",   # near-black
}

# ── Load data ─────────────────────────────────────────────────────────────────
df      = pd.read_csv("data/tpae_properties.csv")
df_fine = pd.read_csv("data/tpae_properties_fine.csv")
metrics = pd.read_csv("data/model_metrics.csv")

targets = ["Tg_K", "Tm_K", "E_MPa"]
ylabels = {
    "Tg_K":  "Glass Transition Temp (K)",
    "Tm_K":  "Melting Temp (K)",
    "E_MPa": "Elastic Modulus (MPa)",
}
fine_cols = {"Tg_K": "Tg_K", "Tm_K": "Tm_K", "E_MPa": "E_MPa"}

# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: Property trends vs composition
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
fig.suptitle("TPAE Properties vs. PTMO Content\n(Group Contribution Theory + Simulated Experimental Data)",
             fontsize=12, y=1.01)

for ax, target in zip(axes, targets):
    ax.plot(df_fine["PTMO_wt_frac"] * 100,
            df_fine[fine_cols[target]],
            color=COLORS["physics"], lw=2, label="Group contribution (physics)")
    ax.scatter(df["PTMO_wt_frac"] * 100,
               df[target],
               color=COLORS["data"], zorder=5, s=60, label="Simulated experiments")
    ax.set_xlabel("PTMO Content (wt%)")
    ax.set_ylabel(ylabels[target])
    ax.set_title(target.replace("_", " "))
    ax.legend(fontsize=8)

    # Mark the 6 experimental compositions
    for x in df["PTMO_wt_frac"] * 100:
        ax.axvline(x, color="gray", lw=0.5, alpha=0.4, ls="--")

plt.tight_layout()
plt.savefig("figures/fig1_property_trends.png", bbox_inches="tight")
plt.close()
print("Saved figures/fig1_property_trends.png")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: Model comparison — LOOCV predictions vs true
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
fig.suptitle("LOOCV Predictions: Physics vs. Baseline NN vs. Physics-Guided NN",
             fontsize=12, y=1.01)

model_styles = [
    ("y_physics",  "Physics only",     COLORS["physics"],  "--", "o"),
    ("y_baseline", "Baseline NN",      COLORS["baseline"], ":",  "s"),
    ("y_guided",   "Physics-guided NN", COLORS["guided"],  "-",  "^"),
]

for ax, target in zip(axes, targets):
    pred_df = pd.read_csv(f"data/predictions_{target}.csv")
    x = pred_df["PTMO_wt_frac"] * 100

    ax.plot(x, pred_df["y_true"],
            "o", color=COLORS["data"], ms=8, zorder=10, label="Simulated data", mew=1.5)

    for col, label, color, ls, marker in model_styles:
        ax.plot(x, pred_df[col], ls=ls, color=color,
                marker=marker, ms=5, lw=1.8, label=label)

    ax.set_xlabel("PTMO Content (wt%)")
    ax.set_ylabel(ylabels[target])
    ax.set_title(target.replace("_", " "))
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("figures/fig2_model_comparison.png", bbox_inches="tight")
plt.close()
print("Saved figures/fig2_model_comparison.png")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: MAE bar chart — model comparison across targets
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
fig.suptitle("LOOCV Mean Absolute Error by Model and Target Property",
             fontsize=12, y=1.01)

bar_models  = ["Physics only", "Baseline NN", "Physics-guided NN"]
mae_keys    = ["mae_physics", "mae_baseline", "mae_guided"]
bar_colors  = [COLORS["physics"], COLORS["baseline"], COLORS["guided"]]
units       = {"Tg_K": "K", "Tm_K": "K", "E_MPa": "MPa"}

x_pos = np.arange(len(bar_models))
for ax, target in zip(axes, targets):
    row = metrics[metrics["target"] == target].iloc[0]
    vals = [row[k] for k in mae_keys]
    bars = ax.bar(x_pos, vals, color=bar_colors, width=0.5, edgecolor="white")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(vals) * 0.01,
                f"{v:.1f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(["Physics", "Baseline\nNN", "Physics-\nguided NN"], fontsize=9)
    ax.set_ylabel(f"MAE ({units[target]})")
    ax.set_title(target.replace("_", " "))
    ax.set_ylim(0, max(vals) * 1.2)

plt.tight_layout()
plt.savefig("figures/fig3_mae_comparison.png", bbox_inches="tight")
plt.close()
print("Saved figures/fig3_mae_comparison.png")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 4: Small-data problem illustration (why physics matters)
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4.5))

# Conceptual: R² vs n_samples for physics vs baseline vs guided
n_samples = np.array([3, 4, 5, 6, 10, 20, 50, 100])
r2_baseline = np.array([-4.0, -1.5, -0.3, 0.5, 0.75, 0.88, 0.93, 0.96])
r2_physics  = np.array([0.94, 0.94, 0.94, 0.96, 0.96, 0.96, 0.96, 0.96])
r2_guided   = np.array([0.70, 0.82, 0.89, 0.92, 0.95, 0.97, 0.97, 0.97])
r2_baseline = np.clip(r2_baseline, -1.5, 1.0)

ax.axvline(6, color="gray", ls="--", lw=1.2, alpha=0.6, label="Our dataset (n=6)")
ax.plot(n_samples, r2_physics,  color=COLORS["physics"],  lw=2.5, marker="o", ms=5, label="Physics only")
ax.plot(n_samples, r2_baseline, color=COLORS["baseline"], lw=2.5, marker="s", ms=5, label="Baseline NN")
ax.plot(n_samples, r2_guided,   color=COLORS["guided"],   lw=2.5, marker="^", ms=5, label="Physics-guided NN")
ax.axhline(0, color="black", lw=0.8, ls=":")

ax.set_xlabel("Number of training samples")
ax.set_ylabel("R² score (LOOCV)")
ax.set_title("Why Physics Priors Matter in Small-Data Regimes\n(Tg prediction, illustrative)")
ax.legend()
ax.set_xlim(2, 105)
ax.set_xscale("log")

plt.tight_layout()
plt.savefig("figures/fig4_small_data_motivation.png", bbox_inches="tight")
plt.close()
print("Saved figures/fig4_small_data_motivation.png")

print("\nAll figures saved to figures/")
