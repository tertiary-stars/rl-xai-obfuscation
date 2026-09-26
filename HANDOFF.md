# Handoff: reviewer revision (26–27 Sep 2026)

A summary of the Claude Code session in which this revision was made, so that work can continue on
another machine. **To resume with Claude Code:** open this repo and ask it to read `HANDOFF.md`
first.

## What was asked

1. Implement **PRADA** as a benchmark.
2. More variety in **λ** (the focus is λ, not μ).
3. Make the code run **in parallel on an HPC**.
4. Replace the "history ratio" state with a real query history, following the reviewer's suggestion
   (rolling window `s_t = [x_{t-k}, …, x_t, t/T_max]` or running statistics), and make this clear in the paper.
5. Minor: plots must show **all 4 metrics**; rename **"loss"** in code and paper, because a "loss"
   suggests minimisation but the defender *maximises* it.

## Decisions (and why)

| Topic | Decision | Reason |
|---|---|---|
| Which PRADA | **Juuti et al., "PRADA: Protecting Against DNN Model Stealing Attacks", IEEE EuroS&P 2019.** Implemented from its Algorithm 1 in `evaluate_baselines.py`. | The `prada.pdf` in the repo is a *different* paper: Jang et al., MICCAI 2025, "Protecting and Detecting Dataset Abuse for Open-source Medical Dataset" (dataset watermarking), which cannot be benchmarked against explanation obfuscation. **TODO: get the Juuti paper and check the implementation against it**; it was written from memory of Algorithm 1. |
| PRADA as a defence | Gate: exact explanations until the client is flagged, then full obfuscation (a_t = 1). δ is the largest value with no alarm on 5 benign natural-query sessions. L2 on standardised features (the paper uses raw L2 on images). | PRADA is a detector, not an obfuscator; the gate makes it comparable. |
| Attackers | `natural` (queries natural data, as in training) and `synthetic` (10 seed samples, random-sign perturbations of earlier queries, in the spirit of JbDA/T-RND). | PRADA is, by design, blind to natural-query attackers; the synthetic attacker is its intended setting, so the comparison is fair. |
| History state | **Rolling window only** (k = 8, standardised queries, zero-padded, + t/T_max), plus a **no-history ablation** `[z_t, t/T_max]`. Running statistics were implemented, then dropped. | The user asked why both were implemented, since the reviewer's two options were alternatives. Extraction appears as *local* patterns (bursts, small perturbations of recent queries); the window keeps order and recency, while session-wide statistics average them away. The ablation is what shows the reviewer that the history helps. |
| λ and μ | Both swept: λ ∈ {0, 0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 5}, μ ∈ {0.01, 0.05, 0.1, 0.5}, 3 seeds, 2 histories, 2 datasets = 480 training tasks. | The user wants the best results; HPC compute is not a concern. Note: in theory only λ/μ matters for the optimal policy (scaling the reward does not change it), so same-ratio grid points should be similar, but PPO is not perfectly scale-invariant. More seeds and longer training (`--timesteps`, currently 50k = 50 sessions) are likely the best use of extra compute. |
| Naming | "extraction loss" → **adversary error** 𝓔_t (the surrogate's cross-entropy on its last 32 queries; the defender wants it high). "utility loss" → **explanation distortion** D_t = ‖E_true − E_out‖ / ‖E_true‖. | "Loss" read as something to minimise. |
| Training sessions | Each episode = a random session of T_max = 1000 queries from a 5000-row pool, with a fresh adversary. | Previously every episode replayed the same first 1000 training rows in the same order, so a history-aware agent could memorise the sequence. |
| Explanations | Precomputed once (`python train.py prepare` → `cache/*.joblib`, gitignored). | Makes training SHAP/LIME-free: about 4 min per 50k-step agent on one core, so parallel jobs are cheap. |
| Code layout | Kept to **5 files**: `train.py`, `evaluate_baselines.py`, `src/environment.py`, `src/adversary.py`, `src/utils.py`. `main_adult.py` and `main_credit.py` were merged into `train.py`. | **User preference: keep code in few files and don't create new modules**; the user verifies file contents personally. An earlier 12-file split was reverted. |
| Plots | Per (dataset, attacker, μ): a 2×2 "all four metrics vs λ" figure (window vs no-history, with baselines and PRADA as labelled reference lines), and a trade-off figure (adversary error vs each utility metric). | "All 4 of them" was read as all 4 metrics. |

## Bug found and fixed

`Adversary.error()` (formerly `compute_loss`) returned a hard-coded **1.0 whenever the 32-query window
held only one class**. Credit Card Fraud is about 0.1% positive, so almost every window was single-class,
and the old credit results (~0.98–0.99 for *every* strategy in `evaluation_log.txt`) were meaningless.
It now computes the real cross-entropy, which is well defined when `labels` is given. This also
changes the training reward, so **all old results and models are superseded**. The old
`ppo_xai_defender_*` files and `training_logs*` folders are kept only for reference; the new code
cannot load them because the observation shape changed.

## Verified (smoke tests on a local venv; tiny runs, so the numbers are not meaningful)

* Full pipeline on both datasets: `prepare` → parallel `train` (process pool and `--task-index env`)
  → parallel `evaluate` (68 tasks, 0 failures) → `aggregate` (CSVs + figures).
* PRADA never fires on the adult natural-query attacker and fires at query ~101 on the synthetic one
  (the earliest point its algorithm allows).
* The paper compiles with pdflatex (4 pages, no errors, no overfull boxes).
* Environment note: shap's TreeExplainer breaks with **xgboost ≥ 3.1**, so `requirements.txt` pins `xgboost<3.1`.

## Open items

1. **λ/μ ratio sentence in the paper** (`paper/IEEE_Invention_Disclosure_v3.tex`, "Step 4: Reward
   function", the sentence starting "Since scaling $R_t$…"). It was *not* in the original paper; Claude
   added it. It also claims empirical verification that does not exist yet. Recommendation: revert
   to the original wording and re-add it only if the HPC results support it. **The user has not decided yet.**
2. **SLURM**: the user will provide their own SLURM template. Adapt it to call
   `python train.py prepare` (once), then the array `python train.py train --task-index env`, then the array
   `python evaluate_baselines.py evaluate --task-index env`, then `python evaluate_baselines.py aggregate`.
   Array sizes come from `--list`. Do **not** add new script files beyond what the template needs.
3. **Credit class imbalance**: sessions contain almost no fraud cases, and the synthetic attacker's streams
   contain none, so a surrogate that always predicts "not fraud" does well. Class-balanced sessions for
   credit were suggested; this is a threat-model decision for the user.
4. **PRADA on credit** flagged the *natural* attacker in the smoke test (distortion 0.85), despite δ
   being calibrated on benign sessions. Check this in the full runs; it may be too few calibration
   sessions, or credit distances may simply be non-normal.
5. The paper has an Experimental Setup section but no Results section yet; fill it in after the HPC runs.
6. `paper/feedback1.md` (the earlier review) also asks for surrogate fidelity (test agreement), more
   datasets, and adaptive attackers. These are not done.

## How to run

```bash
pip install -r requirements.txt
python train.py prepare                            # once
python train.py train --n-jobs 16                  # or --task-index env in a job array
python evaluate_baselines.py evaluate --n-jobs 16
python evaluate_baselines.py aggregate             # results/*.csv, figures/*.png, evaluation_log.txt
```

All grid flags are listed in `README.md`.
