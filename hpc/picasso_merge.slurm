#!/bin/bash
#SBATCH --job-name=alj-merge
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH --output=logs/merge_%j.out
#SBATCH --error=logs/merge_%j.err

set -euo pipefail
module load python/3.11 || true
source "$HOME/venv-alj/bin/activate"
cd "$SLURM_SUBMIT_DIR/../drivers"
python3 merge_tables.py
