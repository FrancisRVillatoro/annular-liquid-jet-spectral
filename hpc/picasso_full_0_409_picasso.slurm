#!/bin/bash
#SBATCH --job-name=alj_full
#SBATCH --array=0-409%138
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=04:00:00
#SBATCH --constraint=cpu
#SBATCH --output=logs/full_%A_%a.out
#SBATCH --error=logs/full_%A_%a.err

# Final clean reproducibility run: all 410 registered tasks, indices 0--409.
# Clear data/partial before submitting this array.  Merge only afterok.

set -eo pipefail

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

module purge
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER
unset PYTHONHOME PYTHONPATH
module load python/3.11.4
hash -r

PY=/mnt/home/soft/python/programs/x86_64/python_3.11.4/bin/python
test -x "$PY"

cd "$SLURM_SUBMIT_DIR/../drivers"

printf -v IDX "%03d" "$SLURM_ARRAY_TASK_ID"

echo "HOST=$(hostname)"
echo "ARRAY_INDEX=$IDX"
echo "START=$(date -Is)"

srun "$PY" run_task.py --run "$SLURM_ARRAY_TASK_ID"

echo "END=$(date -Is)"
