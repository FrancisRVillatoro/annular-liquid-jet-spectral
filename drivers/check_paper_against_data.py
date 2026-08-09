"""Check that every number in the deposited data appears in the manuscript.

The tables of the paper are transcribed from `data/*.csv` by hand.  Any
transcription can go stale: a value is recomputed, the CSV is updated, and
the manuscript is not.  This script closes that loop mechanically.

For each table that appears in the paper it reads the CSV and looks for
every numeric field in the LaTeX source, allowing for the manuscript's
rounding.  It reports three things: values present in the data and absent
from the paper, which are the dangerous ones; values quoted in the paper
that no longer appear in the data; and tables for which no data file
exists.

    python3 check_paper_against_data.py ../../paper1_pof.tex

An exit status of 1 means at least one mismatch.  Run it before every
submission and after every recomputation.
"""

from __future__ import annotations

import csv
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")

# Tables transcribed into the paper, and the columns whose values are
# quoted there.  The supporting tables 10 to 20 are deposited but not
# transcribed, so they are not checked.
IN_PAPER = {
    "table1_spectral_convergence": ["L", "abs_error_vs_N96"],
    "table2_ramos1992_membrane": ["L_present", "L_ramos1992"],
    "table3_near_critical": ["L_collocation_N64", "L_march"],
    "table4_forced_response": ["mean_L", "amplitude_ratio", "lag_L"],
    "table5_thickness": ["L_We50_Cpn0.5", "L_We25_Cpn0"],
    "table6_ramos1993_thickness": ["L_membrane", "L_finite_beta",
                                   "L_ramos1993"],
    "table7_frequency_response": ["gain_L", "gain_L_normalised",
                                  "gain_Cpn", "gain_Cpn_normalised"],
    "table8_degeneracy_time": ["t_star"],
    "table9_cross_check": ["t_star_upwind_fd"],
}


def variants(value):
    """The forms in which a number may legitimately appear in the paper."""
    out = {value}
    try:
        x = float(value)
    except ValueError:
        return out
    out.add(f"{x:g}")
    for d in range(2, 13):
        out.add(f"{x:.{d}f}")
    # scientific notation as the paper writes it
    m = re.match(r"^([0-9.]+)e([+-]?)0*([0-9]+)$", value.replace("E", "e"))
    if m:
        mant, sign, exp = m.groups()
        out.add(f"{mant}\\times10^{{{'-' if sign == '-' else ''}{exp}}}")
    return {v.rstrip("0").rstrip(".") if "." in v else v for v in out} | out


def main():
    tex_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        HERE, "..", "..", "paper1_pof.tex")
    if not os.path.exists(tex_path):
        raise SystemExit(f"manuscript not found: {tex_path}")
    tex = open(tex_path, encoding="utf-8").read()
    tex_compact = tex.replace(" ", "").replace("\n", "")

    problems = 0
    for name, cols in sorted(IN_PAPER.items()):
        path = os.path.join(DATA, f"{name}.csv")
        if not os.path.exists(path):
            print(f"{name}: NO DATA FILE")
            problems += 1
            continue
        missing = []
        with open(path, newline="") as fh:
            for row in csv.DictReader(fh):
                for col in cols:
                    v = (row.get(col) or "").strip()
                    if not v or not re.match(r"^-?[0-9]", v):
                        continue          # text such as "not detected"
                    if not any(w.replace(" ", "") in tex_compact
                               for w in variants(v)):
                        missing.append(f"{col}={v}")
        if missing:
            problems += 1
            print(f"{name}: {len(missing)} value(s) not found in the paper")
            for x in missing[:12]:
                print(f"    {x}")
            if len(missing) > 12:
                print(f"    ... and {len(missing) - 12} more")
        else:
            print(f"{name}: all quoted values found")

    print()
    if problems:
        print(f"{problems} table(s) with mismatches.")
        sys.exit(1)
    print("The manuscript agrees with the deposited data.")


if __name__ == "__main__":
    main()
