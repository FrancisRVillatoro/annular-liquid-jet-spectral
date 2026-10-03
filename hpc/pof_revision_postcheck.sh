#!/bin/bash
ROOT="${1:-$PWD}"
cd "$ROOT" || { echo "ERROR: cannot cd to $ROOT"; false; }

echo "===== EXPECTED REVISION CSV FILES ====="
for f in \
  data/revision_global_admissibility_precise.csv \
  data/revision_global_admissibility_fd.csv \
  data/revision_touchdown_characterization.csv \
  data/revision_offgrid_touchdown.csv \
  data/revision_temporal_convergence.csv \
  data/revision_A045_periodicity.csv \
  data/revision_linear_frequency_response.csv \
  data/revision_frequency_admissibility.csv \
  data/revision_forced_reproducibility.csv \
  data/revision_all_event_cases.csv \
  data/revision_finite_beta_absorption.csv
do
  if [ -s "$f" ]; then
    echo "OK $f"
  else
    echo "MISSING $f"
  fi
done

echo "===== LOG RETURN CODES ====="
grep -H "PYTHON_RC=" logs/rev2_*.out 2>/dev/null | sort -V || true

echo "===== NONZERO RETURN CODES ====="
grep -H "PYTHON_RC=" logs/rev2_*.out 2>/dev/null | grep -v "PYTHON_RC=0" || true

echo "===== ERROR LOGS WITH CONTENT ====="
find logs -maxdepth 1 -name 'rev2_*.err' -type f -size +0c -print 2>/dev/null | sort -V
