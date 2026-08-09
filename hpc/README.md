# Running the table computations on a cluster

The twenty tables decompose into 394 independent row computations: 76 for
the nine tables of the paper, 180 for two dense sweeps that support claims
in the text, and 138 for a set of studies whose purpose is to settle, in
advance, the questions a referee would reasonably ask about claims that
rest on a single parameter value or on a coarse sample. `drivers/table_tasks.py` is the single registry of what those
rows are; `drivers/run_task.py` runs one of them and writes a one-line CSV
into `data/partial/`; `drivers/merge_tables.py` assembles `data/*.csv` from
those pieces in registry order, so the merged tables do not depend on the
order in which the tasks finished.

The serial path, `drivers/make_tables.py`, calls exactly the same functions.
The two cannot drift apart.

## What parallelism buys, and what it does not

There are 76 tasks, so at most 76 cores do useful work; the rest sit idle.
The wall time is set by the **longest single task**, not by the number of
cores, and no task can be split further: each is one time integration of a
stiff-ish ODE system, which is inherently sequential in time.

The longest tasks are, in order: the finest mesh of the finite-difference
cross-check (`table9`, M = 640), the lowest Strouhal number of the frequency
sweep (`table7`, St = 0.005, which integrates to t = 1600), and the finest
resolution of the degeneracy study (`table8`, N = 96). Expect a wall time of
roughly half an hour plus queueing, against about three hours serial. The
total core time is unchanged.

Per-core speed on a cluster node is typically no better than on a current
laptop, and often slightly worse. The gain here is concurrency, nothing
else.

## Setup, once

```bash
module avail python                 # find the available interpreter
module load python/3.11             # adjust to what is actually there
python3 -m venv $HOME/venv-alj
source $HOME/venv-alj/bin/activate
pip install -r requirements.txt
```

Then edit `hpc/picasso_tables.slurm`: set the queue or partition to what
`sinfo` reports, and adjust the `module load` line to match.

## Running

```bash
cd hpc
sbatch picasso_tables.slurm            # indices   0- 75, the paper tables
sbatch picasso_extra.slurm             # indices  76-255, supporting sweeps
sbatch picasso_audit.slurm             # indices 256-393, supporting studies
sbatch --dependency=afterok:<JOB1>:<JOB2>:<JOB3> picasso_merge.slurm
```

Each of the three has a `_picasso` counterpart carrying the module and
interpreter settings that work on the SCBI machine at Malaga; use those
there, and the generic ones elsewhere after adjusting the `module load`
line and the queue.

The two arrays are independent and can run at the same time. Task indices
are fixed by `TABLE_ORDER` in `table_tasks.py`, and each task writes its own
file in `data/partial/`, so nothing collides.

The dependency makes the merge wait for every array task, so a partial run
never produces a partial table: `merge_tables.py` refuses to write a table
whose rows are not all present and says which are missing.

To re-run only what failed, submit the array again restricted to those
indices, for example `sbatch --array=3,17,42 picasso_tables.slurm`. Tasks
that already have a partial file simply overwrite it with the same value.

## Checking

```bash
cd drivers
python3 run_task.py --list          # 394, with the parameters of each task
ls ../data/partial | wc -l          # should reach 394
python3 merge_tables.py
python3 analyse_beta_law.py         # fits the exponent of the small-beta law
```

A single task can always be run interactively for debugging:

```bash
python3 run_task.py --run 0
```

## Threading

The Slurm script sets `OMP_NUM_THREADS=1` and its equivalents. This matters:
without it, each array task would ask the BLAS threading layer for every
core on the node, and tasks sharing a node would contend for the same cores
and finish slower than a single task alone. The computations are dominated
by ODE right-hand-side evaluations, not by dense linear algebra, so nothing
is lost by running single-threaded.

## Reproducibility

Every task is deterministic and independent of the others, of the array
index, and of the number of tasks running concurrently. Running the same
task twice gives bit-identical output. The merged tables are therefore
identical to those produced by the serial path.
