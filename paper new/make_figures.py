"""Paper figures from the 50k-step grid (altay_run2): Fig. 4 trade-off per dataset, Fig. 5 lambda sweep."""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RUN = sys.argv[1]
OUT = sys.argv[2]
INK, INK_2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
HIST = {"window": dict(color="#2a78d6", marker="o", ls="-", label="RL, rolling-window state"),
        "none": dict(color="#eb6834", marker="s", ls="--", label="RL, no history (ablation)")}
REF_MARK = {"No Defense": "o", "Top-K (k=3)": "^", "Gaussian Noise (lvl=0.5)": "v",
            "Precision Reduction (2 dec)": "P", "Random Subset (p=0.5)": "X", "PRADA": "D"}
REF_NAME = {"No Defense": "No defence", "Top-K (k=3)": "Top-$k$ ($k$=3)", "Gaussian Noise (lvl=0.5)": "Gaussian noise",
            "Precision Reduction (2 dec)": "Precision reduction", "Random Subset (p=0.5)": "Random subset", "PRADA": "PRADA"}
DS = {"adult": "Adult Income (XGBoost + SHAP)", "credit": "Credit Card Fraud (DNN + LIME)"}
METRICS = [("adversary_error", "Adversary error $\\uparrow$"), ("distortion", "Explanation distortion $\\downarrow$"),
           ("spearman", "Spearman's $\\rho$ $\\uparrow$"), ("top3", "Top-3 agreement $\\uparrow$")]

plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "legend.fontsize": 7, "font.family": "serif"})


def style(ax):
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK_2, length=2.5)


def load(ds, attack="natural"):
    df = pd.read_csv(f"{RUN}/results/{ds}_{attack}_all_runs.csv")
    return df[df.type != "RL Agent"].set_index("strategy"), df[df.type == "RL Agent"]


# ---- Fig. 4: security vs distortion, lambda = 1 with mu swept (as in the original Pareto figure), per dataset
fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.15))
for ax, ds in zip(axes, DS):
    style(ax)
    ref, rl = load(ds)
    g = rl[rl["lambda"] == 1.0].groupby(["history", "mu"])[["adversary_error", "distortion"]].mean().reset_index()
    for h, kw in HIST.items():
        d = g[g.history == h].sort_values("distortion")
        ax.plot(d.distortion, d.adversary_error, color=kw["color"], ls=kw["ls"], lw=1.4, marker=kw["marker"],
                ms=4.5, mec="white", mew=0.8, label=kw["label"], zorder=3)
        if h == "none":
            for _, r in d.iterrows():
                if r.mu in ((0.0, 0.01, 0.05, 0.1) if ds == "adult" else (0.0, 0.01)):
                    ax.annotate(f"$\\mu$={r.mu:g}", (r.distortion, r.adversary_error), xytext=(3, 4),
                                textcoords="offset points", fontsize=6.2, color=INK_2)
    for name, m in REF_MARK.items():
        r = ref.loc[name]
        ax.scatter(r.distortion, r.adversary_error, marker=m, s=26 if name != "PRADA" else 22,
                   color=INK if name == "PRADA" else MUTED, edgecolors="white", linewidths=0.6, zorder=2,
                   label=REF_NAME[name])
    ax.set_title(DS[ds], loc="left", color=INK, fontweight="bold")
    ax.set_xlabel("Explanation distortion (lower is better)", color=INK_2)
axes[0].set_ylabel("Adversary error (higher is better)", color=INK_2)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.13), labelcolor=INK_2,
           columnspacing=1.2, handletextpad=0.4)
fig.tight_layout(w_pad=2.0)
fig.savefig(f"{OUT}/fig4_tradeoff.pdf", bbox_inches="tight")
plt.close(fig)

# ---- Fig. 5: all four metrics against lambda (mu = 0.05), both datasets, mean +/- std over 3 seeds
MU = 0.05
fig, axes = plt.subplots(2, 4, figsize=(7.16, 3.0))
for row, ds in enumerate(DS):
    ref, rl = load(ds)
    rl = rl[rl.mu == MU]
    lambdas = sorted(rl["lambda"].unique())
    x = np.arange(len(lambdas))
    for col, (m, label) in enumerate(METRICS):
        ax = axes[row, col]
        style(ax)
        for h, kw in HIST.items():
            s = rl[rl.history == h].groupby("lambda")[m].agg(["mean", "std"]).reindex(lambdas)
            ax.fill_between(x, s["mean"] - s["std"], s["mean"] + s["std"], color=kw["color"], alpha=0.15, lw=0)
            ax.plot(x, s["mean"], color=kw["color"], ls=kw["ls"], lw=1.3, marker=kw["marker"], ms=3.2,
                    mec="white", mew=0.6, label=kw["label"], zorder=3)
        ax.axhline(ref.loc["No Defense", m], color=MUTED, ls=":", lw=1.0, zorder=1, label="No defence")
        ax.set_xticks(x[::2])
        ax.set_xticklabels([f"{l:g}" for l in lambdas][::2])
        if row == 0:
            ax.set_title(label, loc="left", color=INK, fontweight="bold")
        else:
            ax.set_xlabel("$\\lambda$", color=INK_2)
    axes[row, 0].set_ylabel(DS[ds].split(" (")[0], color=INK, fontweight="bold")
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.04), labelcolor=INK_2)
fig.tight_layout(h_pad=1.0, w_pad=1.2, rect=(0, 0.04, 1, 1))
fig.savefig(f"{OUT}/fig5_lambda_sweep.pdf", bbox_inches="tight")
plt.close(fig)
print("ok")
