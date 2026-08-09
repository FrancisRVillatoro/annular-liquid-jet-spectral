"""Assemble data/*.csv from the partial results written by run_task.py.

Rows are ordered by their index in the registry, not by completion time, so
the merged tables are identical whatever order the array tasks finished in.
Missing rows are reported and the corresponding table is not written.
"""

import csv
import os

from table_tasks import TABLES

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
PARTIAL = os.path.join(DATA, "partial")


def main():
    for name, spec in sorted(TABLES.items()):
        rows, missing = [], []
        for i in range(len(spec["tasks"])):
            path = os.path.join(PARTIAL, f"{name}__{i:03d}.csv")
            if not os.path.exists(path):
                missing.append(i)
                continue
            with open(path, newline="") as fh:
                rows.append(next(csv.reader(fh)))
        if missing:
            print(f"  {name}: MISSING rows {missing}; not written")
            continue
        out = os.path.join(DATA, f"{name}.csv")
        with open(out, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(spec["header"])
            w.writerows(rows)
        print(f"  wrote data/{name}.csv  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
