#!/bin/bash
#SBATCH --job-name=alj_audit
#SBATCH --array=256-393%138
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=04:00:00
#SBATCH --constraint=cpu
#SBATCH --output=logs/task_%A_%a.out
#SBATCH --error=logs/task_%A_%a.err

# Supporting studies that anticipate referee questions:
#  256-265  published signature at theta0 = +15 deg   (table 12)
#  266-320  amplitude map of the degeneracy          (table 13)
#  321-332  resolution across the jump in amplitude   (table 14)
#  333-372  transversality margin over parameters     (table 15)
#  373-378  quasi-static gains at other parameters    (table 16)
#  379-380  finer resolutions for t*                  (table 17)
#  381-382  finer meshes for the cross-check          (table 18)
#      383  sensitivity to the residual window        (table 19)
#  384-393  the second degeneracy, better diagnosed   (table 20)
#
# Independent of the other two arrays; all three can run at the same time.
# Task indices are fixed by TABLE_ORDER in table_tasks.py and every task
# writes its own file in data/partial/.

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
