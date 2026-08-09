# Picasso sync: Table XXI and final clean run

This sync contains **no `data/` and no `figures/`**.  It is intended to be
extracted over `~/annular-liquid-jet-spectral_pub` after the current data have
been archived.

Changes relevant to execution:

- registers `table21_characteristic_bound` as tasks 394--409;
- `picasso_audit_picasso.slurm` runs only 394--409 (16 tasks);
- adds `picasso_full_0_409_picasso.slurm` for the final clean 0--409 rerun;
- removes the premature `R_eta(1)>=0` exception from the independent upwind FD
  RHS, so the caller's terminal event can localise the Table IX crossing;
- preserves the NumPy 1.x/2.x trapezoid compatibility shim.

The Table XXI grid is
`N = 32,48,64,96` crossed with `eps_m = 1e-2,1e-3,1e-4,1e-5`, at
`a=0.1`, `St=0.5`.  Diagnostics are sampled every 1e-3 time units while the
terminal `min(m)-eps_m=0` event is localised by `solve_ivp`.
