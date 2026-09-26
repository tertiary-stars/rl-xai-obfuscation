# XAI Obfuscation using Reinforcement Learning

An RL agent (PPO) protects a model's explanations (SHAP / LIME) against explanation-aware model
extraction. For every query it picks an obfuscation level `a_t ∈ [0, 1]` from the user's query
history, and the released explanation is

    E_out = (1 - a_t) * E_true + a_t * noise

The agent is rewarded with `R_t = λ · AdversaryError_t − μ · Distortion_t`.

## Terminology (renamed from "loss")

| Name in code / paper | Old name | Meaning | Defender wants |
|---|---|---|---|
| `adversary_error` (𝓔_t) | extraction loss | cross-entropy of the adversary's surrogate against the target model's labels, over its last 32 queries | **high** (surrogate imitates badly) |
| `distortion` (D_t) | utility loss | ‖E_true − E_out‖ / ‖E_true‖ | low |
| `spearman` | | rank correlation of E_true and E_out | high |
| `top3` | | overlap of the top-3 features of E_true and E_out | high |

## The agent's state: fixed-size query history

The state `s_t` is a rolling window of the user's last k standardised queries (`HistoryEncoder` in `src/environment.py`):

| `--histories` | s_t | size |
|---|---|---|
| `window` | `[z_{t-k+1}, …, z_{t-1}, z_t, t/T_max]` | k·d + 1 |
| `none` | `[z_t, t/T_max]`, no history (ablation that shows what the history contributes) | d + 1 |

Each training episode is a random session drawn from a 5000-query pool with a fresh adversary, so
the agent cannot memorise a fixed query order.

## Baselines

* Static: no defence, top-k (k=3), Gaussian noise (0.5σ_E), precision reduction (2 dec), random subset (p=0.5).
* **PRADA** (Juuti et al., EuroS&P 2019), in `evaluate_baselines.py`. This is Algorithm 1 of the
  paper, a Shapiro–Wilk test on the minimum-distance distribution of a client's queries. It is used as a
  gate: exact explanations until the client is flagged, full obfuscation afterwards. δ is
  calibrated so that benign natural-query sessions are never flagged.
* Two attackers (`--attacks`): `natural` (queries natural data, as in training; PRADA cannot see
  this attacker) and `synthetic` (perturbs 10 seed samples, the kind of attacker PRADA was designed to detect).

## Running

```bash
pip install -r requirements.txt

python train.py prepare                            # once: target models + precomputed explanations -> cache/
python train.py train --n-jobs 16                  # one PPO agent per grid point -> runs/
python evaluate_baselines.py evaluate --n-jobs 16  # baselines, PRADA, every agent, both attackers -> results/
python evaluate_baselines.py aggregate             # CSV tables, evaluation_log.txt, figures/
```

Both scripts take the same grid flags (defaults shown), so the same command line selects the same agents:
`--datasets adult credit`, `--lambdas 0 0.1 0.25 0.5 0.75 1 1.5 2 3 5`, `--mus 0.01 0.05 0.1 0.5`,
`--histories window none`, `--state-window 8`, `--seeds 42 43 44`, `--timesteps 50000`,
`--attacks natural synthetic`, `--eval-steps 500`, `--eval-streams 3`.

With these defaults the grid has 2 × 10 × 4 × 2 × 3 = 480 training tasks.
The optimal policy depends only on λ/μ, so grid points with the same ratio should give similar agents. This works as a consistency check.

### Parallel / HPC execution

Every grid point is an independent task:

* **Process pool on one machine:** `--n-jobs N`. Each worker is limited to one BLAS/torch thread.
* **Job arrays (SLURM, PBS, SGE, LSF):** `--task-index <i>` runs only task *i*, and `--task-index env`
  reads `SLURM_ARRAY_TASK_ID` and similar variables. `--list` prints the grid and its size.

```bash
python train.py prepare                   # once, before the arrays (downloads data)
python train.py train --list              # -> "N tasks (job array range: 0-(N-1))"
# array job, one task each:  python train.py train --task-index env
# then:                      python evaluate_baselines.py evaluate --task-index env   (size from --list)
# finally:                   python evaluate_baselines.py aggregate
```

Finished tasks are skipped on re-run (`--overwrite` forces them), so interrupted jobs can simply be resubmitted.
One 50k-step agent takes a few minutes on one core, because explanations are precomputed and no SHAP/LIME runs during training.

## Outputs

* `runs/<dataset>/<history>/lam<λ>_mu<μ>_seed<s>/`: `model.zip`, `config.json`, `monitor.csv`
* `results/<dataset>/<attack>/`: per-task JSON; `results/<dataset>_<attack>_*.csv` holds the aggregated tables
* `figures/<dataset>_<attack>_mu<μ>_lambda_sweep.png`: 2×2, all four metrics vs λ (window vs no history), with baselines and PRADA as reference lines
* `figures/<dataset>_<attack>_mu<μ>_<history>_tradeoff.png`: adversary error against each of the three utility metrics

## Code layout

```
train.py               grid, parallel execution (pool / job array), prepare + train commands
evaluate_baselines.py  static baselines, PRADA, evaluation loop, aggregation, plots
src/environment.py     history encoder (state), obfuscation, adversary session, metrics, Gymnasium environment
src/adversary.py       online MLP surrogate
src/utils.py           dataset loaders, synthetic-query attacker, data cache (precomputed explanations)
```

The `ppo_xai_defender_*` models and `training_logs*` folders in the repository root come from
the previous formulation (state `[x_t, t/T_max]`, fixed query order, λ=1 with μ swept). The new
code cannot load them because the observation shape changed.
