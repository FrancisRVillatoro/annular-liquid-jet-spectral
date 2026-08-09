#!/bin/bash
#SBATCH --job-name=alj_retry
#SBATCH --array=3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --constraint=cpu
#SBATCH --output=logs/retry_%A_%a.out
#SBATCH --error=logs/retry_%A_%a.err

# Task 3 is the M = 640 mesh of the finite-difference cross-check, which
# returned "not detected" on this machine and t* = 10.766778 elsewhere on
# the same input.  The task now distinguishes a genuine absence of the event
# from a failure of the run and reports the reason, so this rerun will say
# which of the two happened.  It takes a couple of seconds.

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

echo "HOST=$(hostname)"
"$PY" -c "import numpy, scipy; print('numpy', numpy.__version__, '| scipy', scipy.__version__)"
"$PY" run_task.py --run 3
