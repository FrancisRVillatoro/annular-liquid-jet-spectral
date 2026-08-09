#!/bin/bash
#SBATCH --job-name=alj_extra
#SBATCH --array=76-255%180
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=04:00:00
#SBATCH --constraint=cpu
#SBATCH --output=logs/task_%A_%a.out
#SBATCH --error=logs/task_%A_%a.err

# Supporting sweeps, not in the paper tables:
#   76-155  dense Strouhal sweep            (table11_frequency_sweep, 80 rows)
#  156-255  small-beta law                  (table10_beta_law,       100 rows)
#
# Independent of the array 0-75 of picasso_tables_picasso.slurm; both can run
# at the same time.  Task indices are fixed by TABLE_ORDER in table_tasks.py
# and every task writes its own file in data/partial/.

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
