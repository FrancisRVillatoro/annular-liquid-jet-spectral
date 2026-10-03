## v1.1.0 — peer-review revision

This release accompanies the revised *Physics of Fluids* manuscript and supersedes v1.0.0 for the revised numerical results.

### Main changes

- Replaces the endpoint-only event interpretation by a global-admissibility diagnostic and locates the first interior contact.
- Adds an independent upwind finite-difference cross-check of that contact.
- Adds local quadratic-contact diagnostics, componentwise off-grid residuals, temporal-tolerance tests and a 50-cycle periodicity test for the non-contact case `A=0.45`.
- Adds linear frequency response and admissible-band checks.
- Corrects the de-aliased enclosed-volume quadrature from `N+1` to `N+2` Gauss--Legendre points.
- Adds a finite-thickness algebraic closure-absorption audit.
- Adds revised Figure 2 and Figure 5 generation/data.
- Records the UMA SCBI Picasso peer-review campaign (Slurm job 2405358; 12/12 tasks completed with zero exit code).

The Zenodo concept DOI remains `10.5281/zenodo.21858484`. Zenodo will assign a new version-specific DOI to this GitHub release.
