# Running the table computations on Picasso / Slurm

`drivers/table_tasks.py` is the single registry.  The final registry contains
**410 independent tasks, indices 0--409**.  `drivers/run_task.py` writes one
one-line CSV per task into `data/partial/`; `drivers/merge_tables.py` assembles
the final `data/*.csv` files in registry order.

The task ranges are:

- 0--75: paper tables 1--9;
- 76--255: dense supporting sweeps (tables 10--11);
- 256--393: audit studies (tables 12--20);
- 394--409: Table XXI, the characteristic-bound test of the unresolved
  `a=0.10, St=0.50` failure.

On Picasso use the `_picasso` scripts and the exact interpreter path already
embedded in them.  BLAS/OpenMP thread counts are fixed to one.

## Stage A: Table XXI

`picasso_audit_picasso.slurm` is deliberately restricted to
`--array=394-409%16`.

```bash
cd ~/annular-liquid-jet-spectral_pub/hpc
sbatch --parsable picasso_audit_picasso.slurm
```

Submit `picasso_merge_picasso.slurm` with `afterok:<JOBID>`.  Review
`data/table21_characteristic_bound.csv`; it must contain 17 lines including
the header.  The a=0.10 branch is not to be given a physical interpretation
until this table has been checked across N and epsilon.

## Stage B: final clean reproducibility run

After Table XXI is accepted, clear **only** the partial row files and rerun
all 410 tasks in the same Picasso environment:

```bash
cd ~/annular-liquid-jet-spectral_pub
rm -rf data/partial
mkdir -p data/partial
cd hpc
sbatch --parsable picasso_full_0_409_picasso.slurm
```

The full script uses `--array=0-409%138`.  Submit the merge with an
`afterok:<JOBID>` dependency.  The merged CSVs from this clean run are the
ones to freeze for the manuscript/deposit.

## Mandatory checks before either stage

```bash
cd ~/annular-liquid-jet-spectral_pub
PY=/mnt/home/soft/python/programs/x86_64/python_3.11.4/bin/python
sha256sum -c SHA256SUMS
find src drivers docs -type f -name '*.py' -print0 | xargs -0 "$PY" -m py_compile
echo "ALL_PYTHON_SYNTAX_EXIT=$?"
bash -n hpc/picasso_audit_picasso.slurm
echo "TABLE21_SLURM_SYNTAX_EXIT=$?"
bash -n hpc/picasso_full_0_409_picasso.slurm
echo "FULL_SLURM_SYNTAX_EXIT=$?"
bash -n hpc/picasso_merge_picasso.slurm
echo "MERGE_SLURM_SYNTAX_EXIT=$?"
"$PY" drivers/run_task.py --list > /tmp/alj_tasks_410.txt
sed -n '1p' /tmp/alj_tasks_410.txt
sed -n '$p' /tmp/alj_tasks_410.txt
```

The first listing line must be `410`; the last task must be index `409`.
Avoid piping `run_task.py --list` directly into `head`, which can produce an
irrelevant `BrokenPipeError` when `head` closes the pipe early.

## Reproducibility

Every task prints SHA256 digests of the imported source modules plus NumPy and
SciPy versions into its job log.  `SHA256SUMS` checks the synchronized source
before execution.  This instrumentation exists because a previous hand patch
on the cluster caused one Table IX run to use an older source copy.

The final freeze is not a mixture of old and new partial results: it is the
single clean 0--409 run described above.
