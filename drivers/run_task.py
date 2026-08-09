"""Run one task of the table registry and write its partial result.

Intended for a cluster job array: one array index, one task, one small CSV
in data/partial/.  `merge_tables.py` then assembles the final tables.

    python3 run_task.py --list          number of tasks and their names
    python3 run_task.py --run K         run task K (0-based)
"""

import argparse
import csv
import hashlib
import os
import sys
import time

import numpy as np
import scipy as sp

from table_tasks import TABLES, TASKS

PARTIAL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                       "data", "partial")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--run", type=int)
    args = ap.parse_args()

    if args.list:
        print(len(TASKS))
        for k, (name, i, fn, a) in enumerate(TASKS):
            print(f"{k:4d}  {name}  row {i}  args={a}")
        return

    if args.run is None:
        ap.error("give --list or --run K")
    if not 0 <= args.run < len(TASKS):
        ap.error(f"--run must be in [0, {len(TASKS) - 1}]")

    name, i, fn, a = TASKS[args.run]

    # Record which sources produced the result.  Version drift between a
    # local checkout and a cluster has already put a wrong entry in a table
    # once: the finite-difference cross-check reported "not detected" on one
    # machine and a clean event time on another, from what were assumed to
    # be the same sources.  Printing the digests into the job log makes that
    # class of mistake visible instead of silent.
    src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
    for mod in sorted(os.listdir(src)):
        if mod.endswith(".py"):
            with open(os.path.join(src, mod), "rb") as fh:
                print(f"src/{mod} sha256 {hashlib.sha256(fh.read()).hexdigest()}")
    print(f"numpy {np.__version__}  scipy {sp.__version__}", flush=True)

    os.makedirs(PARTIAL, exist_ok=True)
    t0 = time.time()
    row = fn(*a)
    dt = time.time() - t0
    path = os.path.join(PARTIAL, f"{name}__{i:03d}.csv")
    with open(path, "w", newline="") as fh:
        csv.writer(fh).writerow(row)
    print(f"task {args.run}: {name} row {i} done in {dt:.1f} s -> {path}",
          flush=True)


if __name__ == "__main__":
    main()
