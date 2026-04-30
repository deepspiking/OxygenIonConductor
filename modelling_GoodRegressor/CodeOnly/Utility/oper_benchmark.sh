#!/bin/bash
#SBATCH -p F16cpu
#SBATCH -N 8
#SBATCH -n 40                  # 8 nodes * 5 tasks/node = 40 tasks
#SBATCH -c 24                  # 128 cores/node / 5 ~= 25.6 -> 24 is safe/efficient
#SBATCH -t 03:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

module load gcc/10.1.0
set -euo pipefail
mkdir -p logs model_logs out

source /home/issp/materiapps/intel/python3/python3vars-3.9.13-1.sh

# --- IMPORTANT: avoid nested parallelism (joblib uses CPUs via --n-jobs) ---
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export KMP_AFFINITY=granularity=fine,compact,1,0

# Julia env (keep if needed for symbolic models)
export PATH=/home/k0736/k073600/.juliaup/bin:$PATH
which julia || true
julia --version || true
export JULIA_DEPOT_PATH=$HOME/.julia_pysr_depot
export PYTHON_JULIAPKG_OFFLINE=yes

# 40 tasks: 8 models x 5 outer folds
srun --nodes=8 --ntasks=40 --ntasks-per-node=5 --cpus-per-task=24 --kill-on-bad-exit=0 bash -lc '
  set -euo pipefail

  MODELS=(ElasticNet Ridge RandomForest MLP XGBoost LightGBM PhySO EQL)

  MID=$(( SLURM_PROCID / 5 ))        # 0..7
  FOLD=$(( SLURM_PROCID % 5 + 1 ))   # 1..5
  MODEL=${MODELS[$MID]}

  echo "NODE=$(hostname) JOB=${SLURM_JOB_ID:-NA} TASK=${SLURM_PROCID} MODEL=${MODEL} FOLD=${FOLD} CPUS=${SLURM_CPUS_PER_TASK:-1}"

  python3 -u benchmark_v12.py \
    --model "${MODEL}" \
    --outer-fold "${FOLD}" \
    --n-jobs "${SLURM_CPUS_PER_TASK:-1}" \
    --regime n500 \
    --out-dir out \
    > "model_logs/${MODEL}_fold${FOLD}_${SLURM_JOB_ID:-NA}_${SLURM_PROCID}.log.txt" 2>&1
'
