"""Frequency response normalised by the correct quasi-static gains.

The two reference gains are the closed-loop ones computed in
`static_gain.py` with the perturbation that the unsteady boundary condition
actually performs:

    dL/du0   = +7.035783   ->  normalised  (dL/du0)/L = 0.560359
    dCpn/du0 = -0.582084

The negative sign of the second is the antiphase of Cpn in the quasi-static
limit: raising u0 lengthens the jet, increases V, and lowers
Cpn = Cpmax (V0/V - 1).

The dynamic gain is reported divided by the static one, so that the St -> 0
limit must be unity in both columns.  A resonance appears as a maximum of
that ratio.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))


import numpy as np

from annular_spectral import Parameters, UnsteadySpectral, solve_steady

G_L = 0.560359          # (dL/du0)/L, lazo cerrado
G_C = 0.582084          # |dCpn/du0|, lazo cerrado
AMP = 0.02
N = 32


def harmonic(t, f, St):
    w = 2.0 * np.pi * St
    M = np.column_stack([np.ones_like(t), np.cos(w * t), np.sin(w * t)])
    c, *_ = np.linalg.lstsq(M, f, rcond=None)
    return c[0], np.hypot(c[1], c[2]), np.arctan2(-c[2], c[1])


base = Parameters(Fr=10, We=50, theta0_deg=0)
w, prob = solve_steady(N, base)

print(f"a = {AMP},  N = {N},  Fr = 10, We = 50, theta0 = 0, ramp = 2")
print(f"{'St':>7}{'T':>8}{'<L>':>13}{'A_L/(a<L>)':>12}{'/G_L':>8}"
      f"{'A_C/a':>10}{'/G_C':>8}{'phase L':>10}{'phase C':>10}{'t end':>9}")

import sys
for St in [float(x) for x in sys.argv[1:]]:
    p = Parameters(Fr=10, We=50, theta0_deg=0, amplitude=AMP, St=St,
                   ramp_cycles=2.0)
    T = p.period
    op = UnsteadySpectral(N, p, dealias=True)
    y0 = op.state_from_steady(w, prob)
    op.volume_reference = op.volume(op.unpack(y0, 0.0)[1], float(y0[-1]))

    NC = max(8, int(np.ceil(70.0 / T)) + 5)
    te = np.linspace(0.0, NC * T, 60 * NC + 1)
    try:
        sol = op.integrate(y0, NC * T, method="DOP853", rtol=1.0e-11,
                           atol=1.0e-13, t_eval=te, max_step=0.05)
    except Exception as exc:
        print(f"{St:7.3f}{T:8.2f}   {type(exc).__name__}: {str(exc)[:44]}")
        continue
    if not sol.success:
        print(f"{St:7.3f}{T:8.2f}   detenida en t = {sol.t[-1]:.3f}")
        continue

    obs = [op.observables(t, sol.y[:, k]) for k, t in enumerate(sol.t)]
    tt = np.array([o["t"] for o in obs])
    LL = np.array([o["L"] for o in obs])
    CC = np.array([o["Cpn"] for o in obs])
    UU = np.array([o["u0"] for o in obs])
    m = tt >= (NC - 3) * T

    _, _, pu = harmonic(tt[m], UU[m], St)
    Lm, AL, pL = harmonic(tt[m], LL[m], St)
    _, AC, pC = harmonic(tt[m], CC[m], St)

    gL = AL / (AMP * Lm)
    gC = AC / AMP
    fL = (pL - pu + np.pi) % (2 * np.pi) - np.pi
    fC = (pC - pu + np.pi) % (2 * np.pi) - np.pi

    print(f"{St:7.3f}{T:8.2f}{Lm:13.7f}{gL:12.5f}{gL / G_L:8.3f}"
          f"{gC:10.5f}{gC / G_C:8.3f}{fL:10.4f}{fC:10.4f}"
          f"{sol.t[-1]:9.1f}")

print()
print("The St -> 0 limit of the normalised columns must be unity.")
print("A maximum of those columns is a resonance; its absence means that")
print("the response is monotone over the band explored and no claim of")
print("resonant amplification can be made for that observable.")
