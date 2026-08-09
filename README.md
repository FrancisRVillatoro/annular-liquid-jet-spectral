# Annular liquid jets: exact constraint absorption and finite-time loss of transversality

Solvers, drivers and data for the paper

> F. R. Villatoro, *Annular liquid jets: exact constraint absorption and
> finite-time loss of transversality*, submitted to Physics of Fluids (2026).

Every number in every table and figure of the paper is produced by the
code in this repository. Nothing is transcribed by hand except the values
tabulated by Ramos that are used for comparison, which are carried
explicitly in the driver that uses them and are labelled as such in the
output.

## What the code does

An annular liquid jet closes on the symmetry axis at a finite distance
from the nozzle. In the one-dimensional model that distance is not a
boundary condition but an algebraic constraint on the state at the free
end. Writing the mean radius as `R = (1 - eta) * S` absorbs that
constraint identically, so the reduced system carries no constraint and
can be discretised by ordinary Chebyshev collocation. The consequences
are worked out in the paper.

## Layout

```
src/       solvers
  annular_spectral.py   Chebyshev collocation, steady and unsteady, membrane
  annular_thickness.py  finite-thickness reference marcher
  offgrid.py            off-grid residual and Chebyshev tail
  upwind_fd.py          independent third-order upwind finite differences
  laval_critical.py     structure of the critical surface of the steady system

drivers/   one script per group of results
  table_tasks.py            registry of the rows of every table, one
                            independent computation each
  analyse_beta_law.py       fits the exponent of the small-beta law
  make_tables.py            serial path: walks the registry, writes data/*.csv
  run_task.py               runs one row of the registry (for a job array)
  merge_tables.py           assembles data/*.csv from the partial results
  set_doi.py                writes the DOIs into every file that quotes them
  check_paper_against_data.py
                            verifies that every deposited number appears in
                            the manuscript; run before every submission
  make_figures.py           regenerates figures/*.pdf and *.png
  verify_steady.py          convergence, published steady data, near-critical
  verify_unsteady.py        discrete equilibrium and semidiscrete spectrum
  verify_dynamic.py         forced response and harmonic metrics
  verify_transversality.py  the finite-time degeneracy and its verification
  verify_crossval.py        spectral against the upwind finite-difference scheme
  static_gain.py            quasi-static gains with the correct perturbation
  freq_response.py          frequency response normalised by those gains
  audit_quad.py             aliasing of the enclosed-volume quadrature
  audit_dealias.py          effect of de-aliasing on the reported results
  audit_offgrid.py          residual and spectral tail along the degeneracy
  audit_lag.py              phase lags unwrapped in the Strouhal number

data/      CSV output, one file per table of the paper
figures/   PDF and PNG output of make_figures.py
docs/      notes on reproduction and a symbolic check
hpc/       Slurm scripts to run the tables as a job array, and notes
```

## Requirements

Python 3.10 or later, with

```
numpy >= 1.24
scipy >= 1.10
matplotlib >= 3.6
sympy >= 1.11        # only for the symbolic check in docs/
```

The results in the paper were obtained with numpy 2.4.4, scipy 1.17.1 and
matplotlib 3.10.8. No compiled extensions and no parallelism are used;
everything runs on one core.

## Reproducing the paper

```bash
pip install -r requirements.txt
cd drivers
python3 make_tables.py            # data/*.csv
python3 make_figures.py           # figures/*.pdf and *.png
```

Approximate single-core cost. Tables 1 to 3, 5 and 6 and Figures 1, 3 and
4 are cheap, seconds to a couple of minutes each. Table 4 and Figure 2
need about ten minutes each, Figure 5 about fifteen, Table 8 about
twenty-five, Table 7 about forty, and Table 9 about two hours, the last
dominated by the finest mesh of the finite-difference scheme. The whole
set is a few hours.

Individual tables and figures can be selected:

```bash
python3 make_tables.py table5_thickness table6_ramos1993_thickness
python3 make_figures.py 2 5
```

The tables decompose into 76 independent row computations, which can be run
concurrently on a cluster; see `hpc/README.md`. That reduces the wall time
to roughly the longest single task, about half an hour, but not below it:
each task is one time integration and cannot be split further.

## Reading the results

Two points of interpretation matter more than the rest.

The residual reported by `offgrid.py` is evaluated on a fine grid at
points where nothing was imposed, and excludes a neighbourhood of the
nozzle, where the boundary values are imposed algebraically and the
equations are not collocated. It is an a posteriori verification, not a
rigorous bound.

The degeneracy time is the instant at which the transversality measure
`S(tau, 1)` falls below a threshold. The threshold cannot be reduced
indefinitely: below about `1e-6` the right-hand side raises on a tentative
integration stage before the step is accepted. The value quoted in the
paper is stable to nine digits over the range of thresholds that can be
reached, which is a statement about the threshold at fixed resolution and
not about accuracy; convergence in the resolution is reported separately.

## Data provenance

Files under `data/` contain, alongside the computed values, the
convergence lengths tabulated in

- J. I. Ramos, Appl. Math. Modelling **16**, 464-475 (1992), Table 1
- J. I. Ramos, Comput. Mech. **11**, 28-64 (1993), Table 4

reproduced for comparison only, in columns whose names identify them.

## Licence

Code is released under the MIT licence (`LICENSE`). The data files under
`data/` and the figures under `figures/` are released under CC BY 4.0
(`LICENSE-DATA`).

## Citing

See `CITATION.cff`.

## Checking that two copies are the same

`SHA256SUMS` lists the digest of every source and driver. Before comparing
results obtained on different machines, check that they were produced by the
same code:

```bash
sha256sum -c SHA256SUMS
```

Every task also prints the digests of the modules it imported, together with
the numpy and scipy versions, into its job log. This is not decoration: a
divergence between a local checkout and a cluster once put a wrong entry in
one of the tables, reported as an absence of an event when it was in fact a
failed run of an older source file.
