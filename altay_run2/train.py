"""Training: prepare the data cache, then train one PPO agent per grid point.

Grid = datasets x lambdas x mus x history encodings x seeds. Every grid point is an independent
task, so it can run in parallel in two ways:

  * one machine, many cores:  python train.py train --n-jobs 16
  * HPC job array:            python train.py train --task-index $SLURM_ARRAY_TASK_ID
                              (`--list` prints the grid and its size)

Tasks whose output already exists are skipped (use --overwrite to redo them), so an interrupted
run can simply be resubmitted.

    python train.py prepare          # once: target models + precomputed explanations -> cache/
    python train.py train [grid flags]
"""
import os

# One BLAS / torch thread per process - otherwise parallel workers oversubscribe the CPU.
# Must be set before numpy / torch are imported.
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import json
import multiprocessing as mp
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from itertools import product

import numpy as np

from src.environment import HISTORY_MODES
from src.utils import ATTACKS, DATASETS, load_cache, prepare

RUNS_DIR = "runs"
DEFAULT_LAMBDAS = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]
DEFAULT_MUS = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5]  # the paper's μ values; λ is swept at each of them
DEFAULT_SEEDS = [42, 43, 44]


# ---------------------------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------------------------
def history_tag(history, k):
    return f"window_k{k}" if history == "window" else history


def run_name(cfg):
    return f"lam{cfg['lambda']:g}_mu{cfg['mu']:g}_seed{cfg['seed']}"


def run_dir(cfg):
    return os.path.join(RUNS_DIR, cfg["dataset"], history_tag(cfg["history"], cfg["state_window"]), run_name(cfg))


def train_configs(args):
    return [
        {
            "dataset": ds, "lambda": lam, "mu": mu, "history": hist, "state_window": args.state_window,
            "seed": seed, "timesteps": args.timesteps, "max_steps": args.max_steps, "adv_window": args.adv_window,
        }
        for ds, hist, mu, lam, seed in product(args.datasets, args.histories, args.mus, args.lambdas, args.seeds)
        if lam > 0 or mu > 0  # λ = μ = 0 gives a zero reward - nothing to learn
    ]


def describe_config(cfg):
    return (f"{cfg['dataset']} {history_tag(cfg['history'], cfg['state_window'])} "
            f"λ={cfg['lambda']:g} μ={cfg['mu']:g} seed={cfg['seed']}")


def grid_parser():
    """Flags shared by train.py and evaluate_baselines.py, so one command line selects the same grid in both."""
    grid = argparse.ArgumentParser(add_help=False)
    grid.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    grid.add_argument("--lambdas", nargs="+", type=float, default=DEFAULT_LAMBDAS, help="security weights λ")
    grid.add_argument("--mus", nargs="+", type=float, default=DEFAULT_MUS, help="distortion weights μ")
    grid.add_argument("--histories", nargs="+", default=list(HISTORY_MODES), choices=HISTORY_MODES,
                      help="state encoding: rolling window of past queries, or none (ablation)")
    grid.add_argument("--state-window", type=int, default=8, help="k, number of queries in the rolling window")
    grid.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    grid.add_argument("--timesteps", type=int, default=50000)
    grid.add_argument("--max-steps", type=int, default=1000, help="T_max, queries per training session")
    grid.add_argument("--adv-window", type=int, default=32, help="queries the adversary error is averaged over")
    grid.add_argument("--attacks", nargs="+", default=list(ATTACKS), choices=ATTACKS, help="(evaluation) attackers")
    grid.add_argument("--eval-steps", type=int, default=500, help="(evaluation) queries per session")
    grid.add_argument("--eval-streams", type=int, default=3, help="(evaluation) sessions per attacker")
    grid.add_argument("--n-jobs", type=int, default=1, help="parallel worker processes")
    grid.add_argument("--task-index", default=None,
                      help="run only this task of the grid (job arrays); 'env' reads SLURM_ARRAY_TASK_ID etc.")
    grid.add_argument("--list", action="store_true", help="print the task grid and exit")
    grid.add_argument("--overwrite", action="store_true", help="redo tasks whose output already exists")
    return grid


# ---------------------------------------------------------------------------------------------
# Parallel execution
# ---------------------------------------------------------------------------------------------
def _init_worker():
    import torch
    torch.set_num_threads(1)


def resolve_task_index(value):
    if value is None:
        return None
    if value == "env":
        for var in ("SLURM_ARRAY_TASK_ID", "PBS_ARRAYID", "PBS_ARRAY_INDEX", "SGE_TASK_ID", "LSB_JOBINDEX"):
            if var in os.environ:
                return int(os.environ[var])
        raise RuntimeError("--task-index env given but no job-array variable is set.")
    return int(value)


def execute(fn, arg_list, n_jobs, task_index=None, describe=str):
    """Run fn(*args) for every args in arg_list - one entry (job array), serially, or in a process pool."""
    if task_index is not None:
        if not 0 <= task_index < len(arg_list):
            raise IndexError(f"task index {task_index} out of range (grid has {len(arg_list)} tasks)")
        arg_list = [arg_list[task_index]]

    if n_jobs <= 1 or len(arg_list) == 1:
        for args in arg_list:
            print(f"[{datetime.now():%H:%M:%S}] {fn(*args)}", flush=True)
        return

    failures = 0
    with ProcessPoolExecutor(max_workers=n_jobs, mp_context=mp.get_context("spawn"), initializer=_init_worker) as pool:
        futures = {pool.submit(fn, *args): args for args in arg_list}
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                msg = fut.result()
            except Exception as exc:  # keep the other tasks running
                failures += 1
                msg = f"FAILED {describe(futures[fut])}: {exc!r}"
            print(f"[{datetime.now():%H:%M:%S}] ({i}/{len(arg_list)}) {msg}", flush=True)
    if failures:
        raise SystemExit(f"{failures} task(s) failed")


def list_tasks(tasks, describe):
    for i, t in enumerate(tasks):
        print(f"{i:5d}  {describe(t)}")
    print(f"{len(tasks)} tasks  (job array range: 0-{len(tasks) - 1})")


# ---------------------------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------------------------
def train_task(cfg, overwrite=False, n_envs=1):
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    from src.environment import XAIObfuscationEnv

    out = run_dir(cfg)
    if os.path.exists(os.path.join(out, "model.zip")) and not overwrite:
        return f"skip (exists) {out}"
    os.makedirs(out, exist_ok=True)

    torch.set_num_threads(1)
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    env_kwargs = {
        "data": load_cache(cfg["dataset"]), "lambda_param": cfg["lambda"], "mu_param": cfg["mu"],
        "history": cfg["history"], "state_window": cfg["state_window"],
        "adv_window": cfg["adv_window"], "max_steps": cfg["max_steps"],
    }
    env = make_vec_env(XAIObfuscationEnv, n_envs=n_envs, env_kwargs=env_kwargs, seed=cfg["seed"], monitor_dir=out,
                       vec_env_cls=SubprocVecEnv if n_envs > 1 else DummyVecEnv)
    agent = PPO("MlpPolicy", env, verbose=0, learning_rate=3e-4, seed=cfg["seed"], device="cpu")
    started = datetime.now()
    agent.learn(total_timesteps=cfg["timesteps"])
    agent.save(os.path.join(out, "model"))
    env.close()

    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump({**cfg, "train_seconds": (datetime.now() - started).total_seconds()}, f, indent=2)
    return f"trained {out}"


def prepare_task(dataset, force):
    prepare(dataset, force=force)
    return f"prepared {dataset}"


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # λ, μ on non-UTF-8 consoles (e.g. Windows)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p_prep = sub.add_parser("prepare", parents=[grid_parser()])
    p_prep.add_argument("--force", action="store_true", help="rebuild the data cache")
    p_train = sub.add_parser("train", parents=[grid_parser()])
    p_train.add_argument("--n-envs", type=int, default=1, help="parallel environments per PPO agent")
    args = parser.parse_args()

    if args.command == "prepare":
        execute(prepare_task, [(ds, args.force) for ds in args.datasets], args.n_jobs)
        return

    tasks = [(cfg, args.overwrite, args.n_envs) for cfg in train_configs(args)]
    describe = lambda t: "train " + describe_config(t[0])
    if args.list:
        return list_tasks(tasks, describe)
    execute(train_task, tasks, args.n_jobs, resolve_task_index(args.task_index), describe)


if __name__ == "__main__":
    main()
