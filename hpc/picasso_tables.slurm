#!/bin/bash
#SBATCH --job-name=alj-tables
#SBATCH --array=0-075
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=04:00:00
#SBATCH --output=logs/task_%A_%a.out
#SBATCH --error=logs/task_%A_%a.err
# --- check the queue name with `sinfo` and add, e.g.:
# SBATCH --constraint=cal
# SBATCH --qos=...

set -euo pipefail

# One thread per task.  Without this, each array task would try to grab
# every core on the node through the BLAS threading layer, and 76 tasks
# sharing a node would contend for the same cores and run slower than one.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

# --- environment.  Check the available module with `module avail python`
# and create the virtual environment once, before submitting:
#     module load python/3.11
#     python3 -m venv $HOME/venv-alj
#     source $HOME/venv-alj/bin/activate
#     pip install -r requirements.txt
module load python/3.11 || true
source "$HOME/venv-alj/bin/activate"

cd "$SLURM_SUBMIT_DIR/../drivers"

printf -v IDX "%03d" "$SLURM_ARRAY_TASK_ID"
echo "host=$(hostname)  array_index=${IDX}  start=$(date -Is)"
srun python3 run_task.py --run "$SLURM_ARRAY_TASK_ID"
echo "end=$(date -Is)"
