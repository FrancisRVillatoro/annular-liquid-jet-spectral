"""Audit A2: off-grid residual and spectral tail along the degeneracy."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np
from annular_spectral import Parameters, solve_steady, UnsteadySpectral
from offgrid import fine_operators, offgrid_residual, modal_tail

TSTAR = 10.766826258   # A=0.5, Stg=0.5, valor convergido

def run(N, dealias=True):
    p = Parameters(Fr=10., We=50., theta0_deg=0., Cpmax=1., pressure_ratio0=1.,
                   amplitude=0., ramp_cycles=0., body_amplitude=0.5, St_g=0.5)
    w, prob = solve_steady(N, p); L0 = prob.unpack(w)[2]
    uns = UnsteadySpectral(N=N, p=p, dealias=dealias)
    y0 = uns.state_from_steady(w, prob)
    uns.volume_reference = uns.volume(np.concatenate([[1.], y0[N:2*N]]), L0)
    ts = [0.0, 2.0, 5.0, 8.0, 10.0, TSTAR-0.5, TSTAR-0.1, TSTAR-0.02]
    s = uns.integrate(y0, ts[-1], method="DOP853", rtol=1e-11, atol=1e-13,
                      max_step=0.02, t_eval=ts)
    ef, P = fine_operators(uns)
    rows=[]
    for k,t in enumerate(s.t):
        r = offgrid_residual(uns, t, s.y[:,k], ef, P)
        tail = modal_tail(uns, s.y[:,k], t)
        St = uns.unpack(s.y[:,k], t)[1][-1]
        rows.append((t, r["max"], max(tail.values()), St))
    return rows

for N in (24, 32, 48, 64):
    print(f"--- N = {N}")
    print(f"   {'t':>10s} {'||res||_inf':>12s} {'cola modal':>11s} {'S(1)':>10s}")
    for t,r,tl,St in run(N):
        print(f"   {t:10.4f} {r:12.3e} {tl:11.2e} {St:10.3e}")
