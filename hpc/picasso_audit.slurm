#!/bin/bash
#SBATCH --job-name=alj-audit
#SBATCH --array=256-393%138
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=04:00:00
#SBATCH --output=logs/task_%A_%a.out
#SBATCH --error=logs/task_%A_%a.err
# --- check the queue name with `sinfo` and add, e.g.:
# SBATCH --constraint=cpu

# Supporting studies that anticipate referee questions:
#  256-265  published signature at theta0 = +15 deg   (table 12)
#  266-320  amplitude map of the degeneracy           (table 13)
#  321-332  resolution across the jump in amplitude   (table 14)
#  333-372  transversality margin over parameters     (table 15)
#  373-378  quasi-static gains at other parameters    (table 16)
#  379-380  finer resolutions for t*                  (table 17)
#  381-382  finer meshes for the cross-check          (table 18)
#      383  sensitivity to the residual window        (table 19)
#  384-393  the second degeneracy, better diagnosed   (table 20)
#
# Independent of the arrays 0-75 and 76-255; all three can run at the same
# time.  Task indices are fixed by TABLE_ORDER in table_tasks.py and every
# task writes its own file in data/partial/.
#
# On the SCBI machine at Malaga use picasso_audit_picasso.slurm instead,
# which carries the module and interpreter settings that work there.

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
