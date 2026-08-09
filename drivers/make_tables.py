"""Regenerate the numerical content of every table in the paper as CSV.

This is the serial path.  It walks the same registry that the cluster path
uses, `table_tasks.TABLES`, so the two cannot produce different numbers.
Nothing is copied from the manuscript; the only external values are the
convergence lengths tabulated by Ramos, which are carried explicitly in
`table_tasks.py` and are identified by column name in the output.

Usage
-----
    python3 make_tables.py                    # every table
    python3 make_tables.py table5_thickness   # selected tables, by name

Cost on one core.  Tables 1, 2, 3, 5 and 6 are cheap, seconds to a couple
of minutes.  Table 4 takes about ten minutes, table 8 about twenty-five,
table 7 about forty, and table 9 about an hour, the last dominated by the
finest mesh.  See hpc/README.md to run the same computations as a job
array.
"""

import csv
import os
import sys
import time

from table_tasks import TABLES

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")


def build(name):
    spec = TABLES[name]
    print(f"{name}: {len(spec['tasks'])} rows")
    rows = []
    for i, (fn, args) in enumerate(spec["tasks"]):
        t0 = time.time()
        rows.append(fn(*args))
        print(f"   row {i:2d}  {time.time() - t0:8.1f} s", flush=True)
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"{name}.csv")
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(spec["header"])
        w.writerows(rows)
    print(f"   wrote data/{name}.csv")


if __name__ == "__main__":
    for name in (sys.argv[1:] or sorted(TABLES)):
        if name not in TABLES:
            raise SystemExit(f"unknown table {name!r}; "
                             f"choose from {sorted(TABLES)}")
        build(name)
