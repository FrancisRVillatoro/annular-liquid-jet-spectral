#!/bin/bash
#SBATCH --job-name=pof_rev2
#SBATCH --partition=cpu_parti
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --array=0-11%6
#SBATCH --output=logs/rev2_%A_%a.out
#SBATCH --error=logs/rev2_%A_%a.err

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
ROOT="$SLURM_SUBMIT_DIR"
mkdir -p "$ROOT/logs" "$ROOT/data"

if [ ! -x "$PY" ]; then
  echo "ERROR: Python not executable: $PY"
  PYTHON_RC=127
else
  cd "$ROOT/drivers" || { echo "ERROR: cannot cd to $ROOT/drivers"; PYTHON_RC=126; }
fi

TASK="${SLURM_ARRAY_TASK_ID}"
SCRIPT=""
case "$TASK" in
  0) SCRIPT="revision_global_admissibility_precise.py" ;;
  1) SCRIPT="revision_global_admissibility_fd.py" ;;
  2) SCRIPT="revision_touchdown_characterization.py" ;;
  3) SCRIPT="revision_offgrid_touchdown.py" ;;
  4) SCRIPT="revision_temporal_and_periodicity.py" ;;
  5) SCRIPT="revision_linear_frequency_response.py" ;;
  6) SCRIPT="revision_frequency_admissibility.py" ;;
  7) SCRIPT="revision_forced_reproducibility.py" ;;
  8) SCRIPT="revision_all_event_cases.py" ;;
  9) SCRIPT="revision_finite_beta_absorption.py" ;;
 10) SCRIPT="audit_quad.py" ;;
 11) SCRIPT="audit_dealias.py" ;;
esac

if [ -z "$SCRIPT" ]; then
  echo "ERROR: unknown array task $TASK"
  PYTHON_RC=125
elif [ -x "$PY" ] && [ -d "$ROOT/drivers" ]; then
  echo "HOST=$(hostname)"
  echo "DATE=$(date -Is)"
  echo "TASK=$TASK"
  echo "SCRIPT=$SCRIPT"
  "$PY" --version
  "$PY" - <<'PYVERS'
import numpy, scipy
print('numpy', numpy.__version__)
print('scipy', scipy.__version__)
PYVERS
  echo "===== SOURCE HASHES ====="
  sha256sum ../src/*.py revision_*.py audit_quad.py audit_dealias.py
  echo "===== RUN ====="
  "$PY" "$SCRIPT"
  PYTHON_RC=$?
  echo "PYTHON_RC=$PYTHON_RC"
  echo "DATE_END=$(date -Is)"
fi
