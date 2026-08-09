"""Insert the Zenodo and journal DOIs into every file that carries them.

Three files in the repository and one line of the manuscript quote a DOI,
and they must agree.  Editing them by hand is how they stop agreeing, so
this does it in one operation and reports what it changed.

    python3 set_doi.py --zenodo 10.5281/zenodo.1234567
    python3 set_doi.py --zenodo 10.5281/zenodo.1234567 \\
                       --paper  10.1063/5.0987654 \\
                       --tex    ../paper1_pof.tex

The Zenodo DOI to use in the manuscript is the *concept* DOI, the one that
always resolves to the latest version, not the DOI of a particular version.
Zenodo only shows it once a first version has been published, so publish
first, read it off the record, and run this afterwards.
"""

from __future__ import annotations

import argparse
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")


def edit(path, subs):
    if not os.path.exists(path):
        print(f"  skipped (absent): {path}")
        return 0
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    n = 0
    for pattern, repl in subs:
        text, k = re.subn(pattern, repl, text)
        n += k
    if n:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
    print(f"  {os.path.relpath(path, ROOT)}: {n} replacement(s)")
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zenodo", help="concept DOI, e.g. 10.5281/zenodo.1234567")
    ap.add_argument("--paper", help="journal DOI, once accepted")
    ap.add_argument("--tex", default=os.path.join(ROOT, "..", "paper1_pof.tex"))
    args = ap.parse_args()
    if not (args.zenodo or args.paper):
        ap.error("give --zenodo, --paper, or both")

    total = 0
    if args.zenodo:
        z = args.zenodo.replace("https://doi.org/", "")
        print(f"Zenodo concept DOI -> {z}")
        total += edit(os.path.join(ROOT, "CITATION.cff"),
                      [(r'doi: "10\.5281/zenodo\.[^"]*"', f'doi: "{z}"'),
                       (r'\s*# TODO: concept DOI', "")])
        total += edit(args.tex,
                      [(r"https://doi\.org/XX\.XXXX/zenodo\.XXXXXXX",
                        f"https://doi.org/{z}"),
                       (r"%\s*\n%\s*TODO: reserve the Zenodo DOI.*?\n(%.*\n)*",
                        "")])
    if args.paper:
        p = args.paper.replace("https://doi.org/", "")
        print(f"Journal DOI -> {p}")
        total += edit(os.path.join(ROOT, "CITATION.cff"),
                      [(r'doi: "10\.1063/[^"]*"', f'doi: "{p}"'),
                       (r'\s*# TODO: on acceptance', "")])
        total += edit(os.path.join(ROOT, ".zenodo.json"),
                      [(r'"identifier": "10\.1063/[^"]*"',
                        f'"identifier": "{p}"')])

    print(f"\n{total} replacement(s) in total.")
    print("Now refresh the manifest:  sha256sum src/*.py drivers/*.py "
          "docs/*.py > SHA256SUMS")
    if total == 0:
        sys.exit("nothing was replaced; check that the placeholders are intact")


if __name__ == "__main__":
    main()
