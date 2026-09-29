#!/bin/bash
#SBATCH -J "rl_xai_job"
#SBATCH -A c00006
#SBATCH -p a100x4q
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH -t 24:00:00
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=gamzeerdgn99@gmail.com

# Full grid: 2 datasets x 10 lambdas x 6 mus x 2 histories x 3 seeds (minus lambda = mu = 0) = 708 PPO agents,
# trained and evaluated in a pool of one worker per CPU. CPU only - no GPU is used.
# Finished runs are skipped, so if the job hits the time limit just submit it again.

# Load the exact Python module available on Altay
module load Python/Python-3.12.4-openmpi-5.0.3-gcc-11.4.0

# Activate the virtual environment of this project
source ~/rl_xai_env/bin/activate

set -eo pipefail
cd ~/rl-xai-obfuscation

# The data cache is built beforehand (compute nodes may have no internet for the OpenML download).
for ds in adult credit; do
    if [ ! -f "cache/${ds}.joblib" ]; then
        echo "cache/${ds}.joblib missing - copy it over or run 'python train.py prepare' on the login node" >&2
        exit 1
    fi
done

N_JOBS=${SLURM_CPUS_PER_TASK:-64}
# Adds per-query timings (prediction, explanation) to the cache if missing; runs alone so timings are clean.
echo "[$(date)] timing"
python train.py prepare --n-jobs 2
echo "[$(date)] training on $N_JOBS workers"
python train.py train --n-jobs "$N_JOBS"
echo "[$(date)] evaluating"
# --overwrite: re-evaluate every agent so all results carry the defence timing.
python evaluate_baselines.py evaluate --n-jobs "$N_JOBS" --overwrite
echo "[$(date)] aggregating"
python evaluate_baselines.py aggregate
echo "[$(date)] done - results/, figures/, evaluation_log.txt"
