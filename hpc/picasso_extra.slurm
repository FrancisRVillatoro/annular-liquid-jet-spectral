#!/bin/bash
#SBATCH --job-name=alj-extra
#SBATCH --array=076-255
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=04:00:00
#SBATCH --output=logs/task_%A_%a.out
#SBATCH --error=logs/task_%A_%a.err
# Tasks 76-155: dense Strouhal sweep (table 11).
# Tasks 156-255: small-beta law (table 10).
# Independent of the array 0-75 of picasso_tables.slurm, which can run at
# the same time: task indices are fixed by TABLE_ORDER in table_tasks.py
# and every task writes its own file in data/partial/.

set -euo pipefail

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

module load python/3.11 || true
source "$HOME/venv-alj/bin/activate"

cd "$SLURM_SUBMIT_DIR/../drivers"

printf -v IDX "%03d" "$SLURM_ARRAY_TASK_ID"
echo "host=$(hostname)  array_index=${IDX}  start=$(date -Is)"
srun python3 run_task.py --run "$SLURM_ARRAY_TASK_ID"
echo "end=$(date -Is)"
