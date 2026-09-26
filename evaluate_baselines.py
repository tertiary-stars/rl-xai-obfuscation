"""Evaluation: static baselines, PRADA and every trained RL agent against two attackers; tables and plots.

    python evaluate_baselines.py evaluate [grid flags] [--n-jobs N | --task-index env]
    python evaluate_baselines.py aggregate [grid flags]

Uses the same grid flags as train.py, so the same command line selects the same agents.
"""
import os

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import glob
import json
import sys
from datetime import datetime
from itertools import product

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import shapiro

from src.environment import AttackSession, HistoryEncoder, obfuscate, spearman_rho, top_k_agreement
from src.utils import DATASETS, load_cache
from train import (describe_config, execute, grid_parser, list_tasks, resolve_task_index, run_dir, run_name,
                   history_tag, train_configs)

RESULTS_DIR = "results"
FIGURES_DIR = "figures"
EVAL_SEED = 1000

# Column names used in all result tables / plots. "Adversary error" replaces the old
# "extraction loss": it is the surrogate's cross-entropy, which the defender wants HIGH.
METRICS = {
    "adversary_error": "Adversary Error (Security) ↑",
    "distortion": "Explanation Distortion ↓",
    "spearman": "Spearman's ρ ↑",
    "top3": "Top-3 Agreement ↑",
}


# =============================================================================================
# PRADA: Protecting Against DNN Model Stealing Attacks (Juuti et al., IEEE EuroS&P 2019), Alg. 1
# =============================================================================================
# PRADA is a stateful, per-client detector. For every query x with predicted class c it computes
# the minimum distance d_min from x to a "growing set" G_c of earlier queries of the same class.
# Benign clients draw queries from a natural distribution and their d_min values are roughly
# normally distributed; extraction attacks that synthesise queries (small perturbations of earlier
# queries) are not. PRADA runs a Shapiro-Wilk test on the d_min values and flags the client once
# the statistic W drops below a threshold delta.
#
#     D <- {}, G_c <- {}, D_{G_c} <- {}, T_c <- 0 for every class c
#     for each query x:
#         c <- F(x)
#         if G_c is empty:
#             G_c <- {x}, D_{G_c} <- {0}, T_c <- 0
#         else:
#             d_min <- min_{y in G_c} dist(y, x);  D <- D + {d_min}
#             if d_min > T_c:
#                 G_c <- G_c + {x};  D_{G_c} <- D_{G_c} + {d_min}
#                 T_c <- max(T_c, mean(D_{G_c}) - std(D_{G_c}))
#         if |D| > 100:
#             D' <- {z in D : |z - mean(D)| < 3 std(D)}      (outlier removal)
#             attack if W(D') < delta
#
# Adaptation to tabular data: dist() is the L2 distance between standardised feature vectors
# (the paper uses raw L2 on images, where all features share a scale).
class PRADA:
    def __init__(self, delta, x_mean, x_std, min_history=100):
        self.delta = delta
        self.x_mean = np.asarray(x_mean, dtype=np.float64)
        self.x_std = np.where(np.asarray(x_std) == 0, 1.0, x_std).astype(np.float64)
        self.min_history = min_history
        self.reset()

    def reset(self):
        self.D, self.G, self.D_G, self.T = [], {}, {}, {}
        self.last_W = np.nan
        self.detected_at = None

    def statistic(self):
        """Shapiro-Wilk W on the outlier-filtered distance set, or NaN while |D| <= min_history."""
        if len(self.D) <= self.min_history:
            return np.nan
        D = np.asarray(self.D)
        mean, std = D.mean(), D.std()
        D_prime = D[(D > mean - 3 * std) & (D < mean + 3 * std)]
        if len(D_prime) < 3 or np.ptp(D_prime) == 0:
            return 0.0  # degenerate (e.g. identical distances) - clearly not normal
        return float(shapiro(D_prime).statistic)

    def observe(self, x, c, t=None):
        """Process one query; returns True if the client is flagged as an attacker at this query."""
        z = (np.ravel(x) - self.x_mean) / self.x_std
        c = int(c)
        if c not in self.G:
            self.G[c], self.D_G[c], self.T[c] = [z], [0.0], 0.0
        else:
            d_min = float(np.min(np.linalg.norm(np.asarray(self.G[c]) - z, axis=1)))
            self.D.append(d_min)
            if d_min > self.T[c]:
                self.G[c].append(z)
                self.D_G[c].append(d_min)
                self.T[c] = max(self.T[c], float(np.mean(self.D_G[c]) - np.std(self.D_G[c])))

        self.last_W = self.statistic()
        flagged = bool(not np.isnan(self.last_W) and self.last_W < self.delta)
        if flagged and self.detected_at is None:
            self.detected_at = t
        return flagged


def prada_delta(data, n_steps, n_streams=5):
    """Largest delta that raises no false alarm on benign sessions (natural training-distribution queries).

    Juuti et al. choose delta so that benign clients are not flagged."""
    min_W = np.inf
    for s in range(n_streams):
        idx = np.random.default_rng(EVAL_SEED + 100 + s).choice(len(data["X_pool"]), size=n_steps, replace=False)
        detector = PRADA(-np.inf, data["x_mean"], data["x_std"])
        for x, c in zip(data["X_pool"][idx], data["y_pool"][idx]):
            detector.observe(x, c)
            if not np.isnan(detector.last_W):
                min_W = min(min_W, detector.last_W)
    return float(min_W)


# =============================================================================================
# Defence strategies: reset() starts a new session, respond() returns the released explanation
# =============================================================================================
class Strategy:
    name, kind = "strategy", "Static Baseline"

    def reset(self):
        pass

    def respond(self, x, y, e_true, t, rng):
        raise NotImplementedError


class NoDefense(Strategy):
    name = "No Defense"

    def respond(self, x, y, e_true, t, rng):
        return e_true.copy()


class TopK(Strategy):
    def __init__(self, k=3):
        self.k, self.name = k, f"Top-K (k={k})"

    def respond(self, x, y, e_true, t, rng):
        if self.k >= len(e_true):
            return e_true.copy()
        top = np.argsort(np.abs(e_true))[-self.k:]
        e_out = np.zeros_like(e_true)
        e_out[top] = e_true[top]
        return e_out


class GaussianNoise(Strategy):
    def __init__(self, e_std, level=0.5):
        self.e_std, self.level, self.name = e_std, level, f"Gaussian Noise (lvl={level})"

    def respond(self, x, y, e_true, t, rng):
        return e_true + self.level * rng.normal(0.0, self.e_std, size=e_true.shape)


class PrecisionReduction(Strategy):
    def __init__(self, decimals=2):
        self.decimals, self.name = decimals, f"Precision Reduction ({decimals} dec)"

    def respond(self, x, y, e_true, t, rng):
        return np.round(e_true, decimals=self.decimals)


class RandomSubset(Strategy):
    def __init__(self, p=0.5):
        self.p, self.name = p, f"Random Subset (p={p})"

    def respond(self, x, y, e_true, t, rng):
        return e_true * (rng.random(e_true.shape) < self.p)


class PRADAGate(Strategy):
    """PRADA in front of the explainer: exact explanations until the client is flagged,
    full obfuscation (a_t = 1) for the rest of the session afterwards."""
    name, kind = "PRADA", "PRADA"

    def __init__(self, delta, x_mean, x_std, e_std):
        self.detector = PRADA(delta, x_mean, x_std)
        self.e_std = e_std
        self.flagged = False

    def reset(self):
        self.detector.reset()
        self.flagged = False

    def respond(self, x, y, e_true, t, rng):
        self.flagged = self.detector.observe(x, y, t) or self.flagged
        return obfuscate(e_true, 1.0, self.e_std, rng) if self.flagged else e_true.copy()


class RLAgent(Strategy):
    kind = "RL Agent"

    def __init__(self, model, encoder, e_std, name):
        self.model, self.encoder, self.e_std, self.name = model, encoder, e_std, name
        self.last_action = np.nan

    def reset(self):
        self.encoder.reset()

    def respond(self, x, y, e_true, t, rng):
        action, _ = self.model.predict(self.encoder.observe(x, t), deterministic=True)
        self.last_action = float(np.clip(action[0], 0.0, 1.0))
        return obfuscate(e_true, self.last_action, self.e_std, rng)


def static_baselines(e_std):
    return [NoDefense(), TopK(3), GaussianNoise(e_std, 0.5), PrecisionReduction(2), RandomSubset(0.5)]


# =============================================================================================
# Evaluation loop
# =============================================================================================
def attack_streams(data, attack, n_steps, n_streams):
    """Query sessions sent by the adversary; identical for every strategy that is evaluated."""
    if attack == "natural":
        for s in range(n_streams):
            idx = np.random.default_rng(EVAL_SEED + s).permutation(len(data["X_eval"]))[:n_steps]
            yield data["X_eval"][idx], data["y_eval"][idx], data["E_eval"][idx]
    else:
        for s in range(min(n_streams, len(data["X_synth"]))):
            yield data["X_synth"][s][:n_steps], data["y_synth"][s][:n_steps], data["E_synth"][s][:n_steps]


def evaluate_strategy(strategy, data, attack, n_steps=500, n_streams=3, adv_window=32):
    per_stream = {m: [] for m in METRICS}
    actions, detections = [], []

    for s, (X, y, E) in enumerate(attack_streams(data, attack, n_steps, n_streams)):
        rng = np.random.default_rng(EVAL_SEED + s)
        session = AttackSession(data["adv_mean"], data["adv_std"], adv_window)
        strategy.reset()
        steps = {m: [] for m in METRICS}
        for t in range(len(X)):
            e_out = strategy.respond(X[t], y[t], E[t], t, rng)
            adversary_error, distortion = session.step(X[t], y[t], E[t], e_out)
            steps["adversary_error"].append(adversary_error)
            steps["distortion"].append(distortion)
            steps["spearman"].append(spearman_rho(E[t], e_out))
            steps["top3"].append(top_k_agreement(E[t], e_out, 3))
            if isinstance(strategy, RLAgent):
                actions.append(strategy.last_action)
        for m in METRICS:
            per_stream[m].append(float(np.mean(steps[m])))
        if isinstance(strategy, PRADAGate):
            detections.append(strategy.detector.detected_at)

    result = {m: float(np.mean(v)) for m, v in per_stream.items()}
    result.update({f"{m}_std": float(np.std(v)) for m, v in per_stream.items()})
    if actions:
        result["mean_action"] = float(np.mean(actions))
    if isinstance(strategy, PRADAGate):
        result["prada_delta"] = strategy.detector.delta
        result["prada_detected_at"] = detections
    return result


def result_path(kind, dataset, attack, cfg):
    base = os.path.join(RESULTS_DIR, dataset, attack)
    if kind == "baselines":
        return os.path.join(base, "baselines.json")
    return os.path.join(base, "rl", history_tag(cfg["history"], cfg["state_window"]), run_name(cfg) + ".json")


def eval_task(kind, dataset, attack, cfg, eval_steps, eval_streams, overwrite=False):
    out = result_path(kind, dataset, attack, cfg)
    if os.path.exists(out) and not overwrite:
        return f"skip (exists) {out}"

    data = load_cache(dataset)
    common = {"n_steps": eval_steps, "n_streams": eval_streams}
    rows = []
    if kind == "baselines":
        strategies = static_baselines(data["e_std"])
        strategies.append(PRADAGate(prada_delta(data, eval_steps), data["x_mean"], data["x_std"], data["e_std"]))
        for strategy in strategies:
            rows.append({"strategy": strategy.name, "type": strategy.kind,
                         **evaluate_strategy(strategy, data, attack, **common)})
    else:
        from stable_baselines3 import PPO

        model_path = os.path.join(run_dir(cfg), "model.zip")
        if not os.path.exists(model_path):
            return f"MISSING model {model_path} - train it first"
        # t/T_max must use the T_max the agent was trained with.
        encoder = HistoryEncoder(data["x_mean"], data["x_std"], t_max=cfg["max_steps"], mode=cfg["history"],
                                 k=cfg["state_window"])
        strategy = RLAgent(PPO.load(model_path, device="cpu"), encoder, data["e_std"],
                           name=f"RL (λ={cfg['lambda']:g}, μ={cfg['mu']:g})")
        rows.append({"strategy": strategy.name, "type": strategy.kind,
                     **{k: cfg[k] for k in ("lambda", "mu", "seed", "history", "state_window")},
                     **evaluate_strategy(strategy, data, attack, adv_window=cfg["adv_window"], **common)})

    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    return f"evaluated {out}"


# =============================================================================================
# Plots - every figure shows all four metrics
# =============================================================================================
INK, INK_2, MUTED, GRID = "#0b0b0b", "#52514e", "#a3a29d", "#e6e5e1"
HISTORY_COLORS = {"window": "#2a78d6", "none": "#eb6834"}
HISTORY_LABELS = {"window": "RL, rolling-window history", "none": "RL, no history (ablation)"}
PRADA_COLOR = "#3b3a37"
REF_MARKERS = ["o", "s", "^", "v", "P", "X", "*"]
LAMBDA_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list("seq_blue", ["#86b6ef", "#0d366b"])


def _style(ax):
    ax.set_facecolor("white")
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK_2, labelsize=9)


def _spread_labels(values, min_gap):
    """Nudge label y-positions apart so reference-line labels do not overlap."""
    order = np.argsort(values)
    placed = np.array(values, dtype=float)
    for prev, cur in zip(order[:-1], order[1:]):
        if placed[cur] - placed[prev] < min_gap:
            placed[cur] = placed[prev] + min_gap
    return placed


def plot_lambda_sweep(rl_df, ref_df, title, path):
    """2x2 grid, one panel per metric: RL agents vs lambda (mean +/- std over seeds), one line per history
    encoding; static baselines and PRADA as labelled horizontal reference lines."""
    lambdas = sorted(rl_df["lambda"].unique())
    pos = {lam: i for i, lam in enumerate(lambdas)}  # lambda grid is non-uniform -> evenly spaced ticks

    fig, axes = plt.subplots(2, 2, figsize=(14, 8.5))
    for ax, (metric, label) in zip(axes.ravel(), METRICS.items()):
        _style(ax)
        for history, grp in rl_df.groupby("history"):
            stats = grp.groupby("lambda")[metric].agg(["mean", "std"]).reindex(lambdas).fillna(0.0)
            x = [pos[l] for l in stats.index]
            color = HISTORY_COLORS.get(history, INK)
            ax.fill_between(x, stats["mean"] - stats["std"], stats["mean"] + stats["std"], color=color, alpha=0.15, linewidth=0)
            ax.plot(x, stats["mean"], color=color, linewidth=2, marker="o", markersize=6,
                    markeredgecolor="white", markeredgewidth=1.5, label=HISTORY_LABELS.get(history, history), zorder=3)

        refs = ref_df.sort_values(metric)
        for _, row in refs.iterrows():
            is_prada = row["type"] == "PRADA"
            ax.axhline(row[metric], color=PRADA_COLOR if is_prada else MUTED,
                       linestyle="-." if is_prada else "--", linewidth=1.4 if is_prada else 1.0, zorder=1)
        ax.set_xticks(range(len(lambdas)))
        ax.set_xticklabels([f"{l:g}" for l in lambdas])
        ax.set_xlim(-0.3, len(lambdas) - 0.7)
        # Direct labels in the right margin, nudged apart so they never overlap.
        y_lo, y_hi = ax.get_ylim()
        for (_, row), ly in zip(refs.iterrows(), _spread_labels(refs[metric].values, 0.055 * (y_hi - y_lo))):
            is_prada = row["type"] == "PRADA"
            ax.annotate(row["strategy"], xy=(1.0, row[metric]), xytext=(1.02, ly),
                        xycoords=("axes fraction", "data"), textcoords=("axes fraction", "data"),
                        fontsize=7.5, va="center", annotation_clip=False,
                        color=INK if is_prada else INK_2, fontweight="bold" if is_prada else "normal")
        ax.set_xlabel("λ (security weight)", color=INK_2, fontsize=10)
        ax.set_title(label, color=INK, fontsize=11, loc="left", fontweight="bold")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles += [plt.Line2D([], [], color=PRADA_COLOR, linestyle="-.", linewidth=1.4),
                plt.Line2D([], [], color=MUTED, linestyle="--", linewidth=1.0)]
    labels += ["PRADA (Juuti et al.)", "Static baselines"]
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False, fontsize=9,
               labelcolor=INK_2, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle(title, color=INK, fontsize=13, fontweight="bold", x=0.01, ha="left")
    fig.subplots_adjust(left=0.06, right=0.86, top=0.9, bottom=0.12, wspace=0.42, hspace=0.38)
    fig.savefig(path, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f" -> saved {path}")


def plot_tradeoff(rl_df, ref_df, title, path):
    """1x3 grid: security (adversary error, y) against each of the three utility metrics (x).
    RL agents (seed-averaged) are coloured by lambda; baselines and PRADA are identified by marker."""
    utility = [m for m in METRICS if m != "adversary_error"]
    rl_mean = rl_df.groupby("lambda")[list(METRICS)].mean().reset_index().sort_values("lambda")
    norm = matplotlib.colors.Normalize(vmin=0, vmax=max(1, len(rl_mean) - 1))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax, metric in zip(axes, utility):
        _style(ax)
        ax.plot(rl_mean[metric], rl_mean["adversary_error"], color=MUTED, linewidth=1.2, zorder=1)
        for i, (_, row) in enumerate(rl_mean.iterrows()):
            ax.scatter(row[metric], row["adversary_error"], s=70, color=LAMBDA_CMAP(norm(i)),
                       edgecolors="white", linewidths=1.5, zorder=3)
        for idx in (0, len(rl_mean) - 1):
            row = rl_mean.iloc[idx]
            ax.annotate(f"λ={row['lambda']:g}", (row[metric], row["adversary_error"]), textcoords="offset points",
                        xytext=(6, 6), fontsize=8, color=INK_2)
        # Baselines are identified by marker shape (legend below) - text labels collide where points cluster.
        for (_, row), marker in zip(ref_df.iterrows(), REF_MARKERS):
            is_prada = row["type"] == "PRADA"
            ax.scatter(row[metric], row["adversary_error"], s=75, marker="D" if is_prada else marker,
                       color=PRADA_COLOR if is_prada else MUTED, edgecolors="white", linewidths=1.2, zorder=2,
                       label=row["strategy"])
        ax.set_xlabel(METRICS[metric], color=INK_2, fontsize=10)
    axes[0].set_ylabel(METRICS["adversary_error"], color=INK_2, fontsize=10)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.45, 0.0), ncol=len(labels), frameon=False,
               fontsize=9, labelcolor=INK_2)

    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=LAMBDA_CMAP)
    cbar = fig.colorbar(sm, ax=axes, fraction=0.02, pad=0.01, ticks=range(len(rl_mean)))
    cbar.ax.set_yticklabels([f"{l:g}" for l in rl_mean["lambda"]])
    cbar.set_label("λ", color=INK_2)
    cbar.outline.set_visible(False)
    fig.suptitle(title, color=INK, fontsize=13, fontweight="bold", x=0.01, ha="left")
    fig.savefig(path, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f" -> saved {path}")


# =============================================================================================
# Commands
# =============================================================================================
def aggregate(args):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    for ds, attack in product(args.datasets, args.attacks):
        base = os.path.join(RESULTS_DIR, ds, attack)
        if not os.path.exists(os.path.join(base, "baselines.json")):
            print(f"[aggregate] no results for {ds}/{attack} - skipping")
            continue
        with open(os.path.join(base, "baselines.json"), encoding="utf-8") as f:
            ref_df = pd.DataFrame(json.load(f))
        rl_rows = []
        for path in glob.glob(os.path.join(base, "rl", "*", "*.json")):
            with open(path, encoding="utf-8") as f:
                rl_rows += json.load(f)
        rl_df = pd.DataFrame(rl_rows)
        pd.concat([ref_df, rl_df], ignore_index=True).to_csv(os.path.join(RESULTS_DIR, f"{ds}_{attack}_all_runs.csv"), index=False)

        pipeline_name = DATASETS[ds][0]
        summary = ref_df[["strategy", "type", *METRICS]].rename(columns=METRICS)
        with open("evaluation_log.txt", "a", encoding="utf-8") as log:
            log.write(f"\n=======================================================\n"
                      f"TIMESTAMP: {datetime.now():%Y-%m-%d %H:%M:%S}\nPIPELINE:  {pipeline_name}   ATTACK: {attack}\n"
                      f"=======================================================\n{summary.to_string(index=False)}\n")
            print(f"\n=== {pipeline_name} / {attack} attacker ===\n{summary.to_string(index=False)}")
            if rl_df.empty:
                continue
            rl_summary = rl_df.groupby(["history", "mu", "lambda"])[list(METRICS)].agg(["mean", "std"])
            rl_summary.to_csv(os.path.join(RESULTS_DIR, f"{ds}_{attack}_rl_summary.csv"))
            log.write(rl_summary.to_string() + "\n")
            print(rl_summary.to_string())

        for mu, rl_mu in rl_df.groupby("mu"):
            title = f"{pipeline_name}, {attack} attacker (μ={mu:g})"
            plot_lambda_sweep(rl_mu, ref_df, title, os.path.join(FIGURES_DIR, f"{ds}_{attack}_mu{mu:g}_lambda_sweep.png"))
            for history, rl_h in rl_mu.groupby("history"):
                plot_tradeoff(rl_h, ref_df, f"{title}, history: {history}",
                              os.path.join(FIGURES_DIR, f"{ds}_{attack}_mu{mu:g}_{history}_tradeoff.png"))


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # λ, μ, ↑ on non-UTF-8 consoles (e.g. Windows)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("evaluate", parents=[grid_parser()])
    sub.add_parser("aggregate", parents=[grid_parser()])
    args = parser.parse_args()

    if args.command == "aggregate":
        return aggregate(args)

    tasks = []
    for ds, attack in product(args.datasets, args.attacks):
        tasks.append(("baselines", ds, attack, None, args.eval_steps, args.eval_streams, args.overwrite))
        tasks += [("rl", ds, attack, cfg, args.eval_steps, args.eval_streams, args.overwrite)
                  for cfg in train_configs(args) if cfg["dataset"] == ds]
    describe = lambda t: f"eval {t[1]}/{t[2]} " + ("baselines + PRADA" if t[0] == "baselines" else describe_config(t[3]))
    if args.list:
        return list_tasks(tasks, describe)
    execute(eval_task, tasks, args.n_jobs, resolve_task_index(args.task_index), describe)


if __name__ == "__main__":
    main()
