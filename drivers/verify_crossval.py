"""Cross-check: spectral collocation against upwind finite differences.

The finite-difference solver is a method of lines in the variables
(m, R, u, v) on a uniform mesh, with third-order upwind-biased differences,
no global basis and no exact absorption of the tip constraint.  It shares
none of the three with the spectral solver, which is what makes the
agreement informative.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import sys

import numpy as np

from annular_spectral import Parameters, solve_steady, UnsteadySpectral
from upwind_fd import UpwindFD


def body(A=0.5, Stg=0.5):
    return Parameters(Fr=10.0, We=50.0, theta0_deg=0.0, Cpmax=1.0,
                      pressure_ratio0=1.0, amplitude=0.0, ramp_cycles=0.0,
                      body_amplitude=A, St_g=Stg)


def fd_run(p, M, eps=1.0e-2, tf=30.0, Nseed=64, impose_tip=True):
    w, prob = solve_steady(Nseed, Parameters(Fr=p.Fr, We=p.We,
                                             theta0_deg=p.theta0_deg,
                                             Cpmax=p.Cpmax,
                                             pressure_ratio0=p.pressure_ratio0))
    op = UpwindFD(M, p, impose_tip=impose_tip)
    y0 = op.from_spectral(w, prob)

    def ev(t, y):
        R = op.unpack(y)[1]
        return -op.d_central(R)[-1] - eps
    ev.terminal = True
    ev.direction = -1
    try:
        sol = op.integrate(y0, tf, events=ev)
    except FloatingPointError as exc:
        return None, None, str(exc)[:40]
    te = sol.t_events[0]
    R = op.unpack(sol.y[:, -1])[1]
    return (float(te[0]) if te.size else None), float(abs(R[-1])), ""


if __name__ == "__main__":
    p = body()
    print("Tip degeneracy, body-force forcing A=0.5, St_g=0.5")
    print("Spectral, same threshold eps=1e-2:  t* = 10.766817  (N=64)")
    print(f"\n{'M':>6}{'t* (UW3)':>16}{'diff vs spectral':>18}"
          f"{'|R(1)| final':>15}")
    for M in [int(x) for x in (sys.argv[1:] or (80, 160, 320, 640))]:
        t, drift, msg = fd_run(p, M)
        if t is None:
            note = msg if msg else "no detecta el suceso"
            print(f"{M:6d}   {note}")
        else:
            print(f"{M:6d}{t:16.6f}{t - 10.766817:18.2e}{drift:15.2e}")
    print(f"\n{'M':>6}{'tip closure':>20}{'t* o fallo':>24}")
    for M in (160, 320):
        for imp in (True, False):
            t, drift, msg = fd_run(p, M, impose_tip=imp)
            lab = "impuesto" if imp else "derivado"
            cell = f"{t:.6f}" if t is not None else (msg or "sin suceso")
            print(f"{M:6d}{lab:>20}{cell:>24}")

    print("\n|R(1)| measures the drift of the tip constraint, which in this")
    print("scheme is preserved only because dL/dt is chosen so that R_t(1) = 0.")
    print("In the spectral solver R = (1-eta) S annihilates it identically.")
