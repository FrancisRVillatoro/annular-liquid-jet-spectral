#!/bin/bash
RAW=$(sbatch --parsable hpc/pof_revision_campaign.slurm)
CLEAN=$(printf '%s\n' "$RAW" | sed -r $'s/\x1B\\[[0-9;]*[[:alpha:]]//g')
JOB=$(printf '%s\n' "$CLEAN" | awk -F';' '$1 ~ /^[0-9]+$/ {print $1}' | tail -n 1)
printf 'SBATCH_RAW=%q\n' "$RAW"
printf 'JOB=%s\n' "$JOB"
if [ -n "$JOB" ]; then
  squeue -l -j "$JOB"
else
  echo "ERROR: no numeric JobID could be extracted"
fi
