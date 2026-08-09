#!/bin/bash
#SBATCH --job-name=alj_merge
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH --constraint=cpu
#SBATCH --output=logs/merge_%j.out
#SBATCH --error=logs/merge_%j.err

set -eo pipefail

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

module purge
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER
unset PYTHONHOME PYTHONPATH
module load python/3.11.4
hash -r

PY=/mnt/home/soft/python/programs/x86_64/python_3.11.4/bin/python

test -x "$PY"

cd "$SLURM_SUBMIT_DIR/../drivers"

"$PY" merge_tables.py
