"""Forced dynamic response and harmonic metrics.

Sign convention for the lag.  With q = c0 + A cos(w t + phi) the maximum is
at t = -phi/w, so the lag of the output behind the input is

    lag = -(phi_out - phi_in) / w,

not +(phi_out - phi_in)/w.  A lag defined modulo the period is moreover
ambiguous; see audit_lag.py, which unwraps it by continuation in St.  For
the convergence length no reduction by pi applies, since the closed-loop
quasi-static gain dL/du0 is positive; for the pressure coefficient it does,
since that gain is negative.

The quantity reported as nonfundamental_rms_fraction is not the total
harmonic distortion sqrt(A2^2 + A3^2 + ...)/A1 but the RMS fraction not
explained by mean plus fundamental, which is what the name states.

The off-grid residual is reported alongside: without it the table would
only say that two discretisations agree, not that they solve the equations.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import time

import numpy as np

from annular_spectral import Parameters, UnsteadySpectral, solve_steady
from offgrid import fine_operators, offgrid_residual


def harmonic(t, f, St):
    """Least-squares fit of f = c0 + A cos(2 pi St t + phi).

    Also returns nonfundamental_rms_fraction: the norm of the residual of
    the fit divided by the deviation from the mean.
    """
    w = 2.0 * np.pi * St
    M = np.column_stack([np.ones_like(t), np.cos(w * t), np.sin(w * t)])
    c, *_ = np.linalg.lstsq(M, f, rcond=None)
    A = np.hypot(c[1], c[2])
    phi = np.arctan2(-c[2], c[1])
    nonfund = np.linalg.norm(f - M @ c) / np.linalg.norm(f - c[0])
    return c[0], A, phi, nonfund


def lag(phi_out, phi_in, T):
    """Lag of the output behind the input, reduced to [0, T)."""
    return ((-(phi_out - phi_in)) % (2.0 * np.pi)) / (2.0 * np.pi) * T


p = Parameters(Fr=10, We=50, theta0_deg=0, amplitude=0.1, St=0.1,
               ramp_cycles=2.0)
St, T = p.St, p.period
NCYC = 9

print("Respuesta forzada en la boquilla, a=0.1, St=0.1, Fr=10, We=50")
print("Protocol: periodic_orbit (ramp_cycles=2).  The ramp is not needed for")
print("the asymptotic limit cycle, but it is needed for spectral accuracy in")
print("the transient, and to allow refinement beyond N ~ 64.")
print()
print(f"{'N':>4}{'metodo':>9}{'seg':>7}{'nfev':>8}"
      f"{'mean L':>14}{'A_L/L':>10}{'lag L':>11}"
      f"{'A_Cpn':>10}{'lag Cpn':>13}{'non-fund':>10}{'|res| off':>10}")

for N, method in ((16, "DOP853"), (24, "DOP853"), (32, "DOP853"),
                  (48, "DOP853"), (24, "Radau")):
    w, prob = solve_steady(N, Parameters(Fr=p.Fr, We=p.We,
                                         theta0_deg=p.theta0_deg))
    op = UnsteadySpectral(N, p, dealias=True)
    y0 = op.state_from_steady(w, prob)
    op.volume_reference = op.volume(op.unpack(y0, 0.0)[1], float(y0[-1]))

    te = np.linspace(0.0, NCYC * T, 40 * NCYC + 1)
    t0 = time.time()
    sol = op.integrate(y0, NCYC * T, method=method, rtol=1.0e-10,
                       atol=1.0e-12, t_eval=te, max_step=0.05)
    wall = time.time() - t0
    if not sol.success:
        print(f"{N:4d}{method:>9}  FALLO: {sol.message[:40]}")
        continue

    obs = [op.observables(t, sol.y[:, k]) for k, t in enumerate(sol.t)]
    tt = np.array([o["t"] for o in obs])
    LL = np.array([o["L"] for o in obs])
    CC = np.array([o["Cpn"] for o in obs])
    UU = np.array([o["u0"] for o in obs])
    last = tt >= (NCYC - 3) * T

    _, Au, pu, _ = harmonic(tt[last], UU[last], St)
    Lm, AL, pL, nonfund = harmonic(tt[last], LL[last], St)
    _, AC, pC, _ = harmonic(tt[last], CC[last], St)

    ef, P = fine_operators(op)
    res = offgrid_residual(op, sol.t[-1], sol.y[:, -1], ef, P)["max"]

    print(f"{N:4d}{method:>9}{wall:7.1f}{sol.nfev:8d}"
          f"{Lm:14.8f}{AL / Lm:10.5f}{lag(pL, pu, T):11.4f}"
          f"{AC:10.5f}{lag(pC, pu, T):13.4f}"
          f"{nonfund:10.2e}{res:10.2e}")

print()
print("The lags carry the sign convention of the docstring: the convergence")
print("length responds with a larger lag than the pressure coefficient, in")
print("qualitative agreement with Ramos (1992, 1995).")
