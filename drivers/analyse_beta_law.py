"""Fit the small-beta law from data/table10_beta_law.csv.

The corollary of the paper states that

    L(0) - L(beta) = c beta^(1/2) + o(beta^(1/2))    as beta -> 0+,

so that dL/dbeta -> -infinity and the membrane is a square-root singular
limit rather than a regular one.  This script measures the exponent instead
of assuming it: it fits log(L(0) - L(beta)) against log(beta) by least
squares over the tabulated range, and reports the exponent, the
coefficient, and the residual of the fit for each of the two families.
"""

import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from table_tasks import thick_L  # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")


def fit(beta, drop, label):
    a, b = np.polyfit(np.log(beta), np.log(drop), 1)
    pred = np.exp(b) * beta ** a
    rel = np.max(np.abs(pred / drop - 1.0))
    print(f"  {label}")
    print(f"     exponent            {a:.6f}    (1/2 expected)")
    print(f"     coefficient c       {np.exp(b):.6f}")
    print(f"     max relative error  {rel:.2e}  over "
          f"{beta.min():.1e} <= beta <= {beta.max():.1e}")
    return a, np.exp(b)


def main():
    path = os.path.join(DATA, "table10_beta_law.csv")
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found; run the table first")
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    beta = np.array([float(r["beta"]) for r in rows])
    d50 = np.array([float(r["L0_minus_L_We50"]) for r in rows])
    d25 = np.array([float(r["L0_minus_L_We25"]) for r in rows])
    order = np.argsort(beta)
    beta, d50, d25 = beta[order], d50[order], d25[order]

    print(f"membrane values: L(0) = {thick_L(10.0, 50.0, 0.0, 0.5):.10f} "
          f"(We=50, Cpn=0.5), {thick_L(10.0, 25.0, 0.0, 0.0):.10f} "
          f"(We=25, Cpn=0)\n")
    fit(beta, d50, "We = 50, Cpn = 0.5")
    fit(beta, d25, "We = 25, Cpn = 0")

    print("\n  The ratio (L(0) - L(beta)) / sqrt(beta) is tabulated in the "
          "last two\n  columns of the CSV; it tends to c as beta -> 0 and "
          "drifts upwards at\n  the top of the range, where the neglected "
          "o(beta^(1/2)) term is felt.")


if __name__ == "__main__":
    main()
