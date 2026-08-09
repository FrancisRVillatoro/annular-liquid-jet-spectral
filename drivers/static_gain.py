"""Quasi-static gain of the convergence length with respect to u0.

Perturbing u0 in a steady solver that normalises the mass flux to m u = 1 is
*not* the operation performed by the unsteady boundary condition, which is

    m(t,0) = 1,   u(t,0) = u0(t),   R(t,0) = 1,   v(t,0) = u0 tan(theta0),

so that the nozzle mass flux F0 = m0 u0 = u0 varies with the perturbation.
The two are different operations, which is why a static gain obtained the
first way does not match the St -> 0 limit of the dynamic gain.

With general F0 = m0 u0 and m = F0/u, the steady membrane system is

    s_z [F0 u - J/We] = q (Cpn R - q^{-1/2})/We - F0 s/(u Fr),
    u_z = 1/(u Fr) - s Y /(We F0),
    Y   = Cpn R - q^{-1/2} + R s_z / q^{3/2},   q = 1 + s^2,  J = R/sqrt(q),

which at F0 = 1 recovers the determinant We u - J.  Note that the critical
condition becomes We F0 u = J, that is We u0^2 = cos(theta0) at the nozzle:
perturbing the nozzle velocity also moves the critical margin.

The relevant comparison is moreover the closed-loop one, since a change in
u0 changes V and with it Cpn = Cpmax (V0/V - 1), which feeds back on L.
Both gains are reported.
"""

from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))



import math

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

# numpy renamed `trapz` to `trapezoid` in version 2.0 and removed the old
# name later; numpy 1.x has only `trapz`.  Binding the available one here
# keeps a single source working on both, which matters because the cluster
# used for the large runs carries numpy 1.24 while the development machine
# carries 2.4.  Hand-patching the file on one of them is what allowed the
# two copies to drift apart in the first place.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz



def march(u0: float, Cpn: float, Fr: float = 10.0, We: float = 50.0,
          theta0_deg: float = 0.0, z_max: float = 400.0):
    """March with DOP853 until R = 0.  Returns (L, V) with V = int_0^L R^2 dz."""
    F0 = u0                                  # m0 = 1
    tan0 = math.tan(math.radians(theta0_deg))

    def rhs(z, y):
        R, u, s = y
        q = 1.0 + s * s
        J = R / math.sqrt(q)
        den = F0 * u - J / We
        sz = (q * (Cpn * R - 1.0 / math.sqrt(q)) / We
              - F0 * s / (u * Fr)) / den
        Y = Cpn * R - 1.0 / math.sqrt(q) + R * sz / q ** 1.5
        uz = 1.0 / (u * Fr) - s * Y / (We * F0)
        return [s, uz, sz]

    def tip(z, y):
        return y[0]
    tip.terminal = True
    tip.direction = -1

    sol = solve_ivp(rhs, (0.0, z_max), [1.0, u0, tan0], method="DOP853",
                    events=tip, rtol=1.0e-12, atol=1.0e-14, max_step=0.25,
                    dense_output=True)
    if not sol.t_events[0].size:
        return float("nan"), float("nan")
    L = float(sol.t_events[0][0])
    z = np.linspace(0.0, L, 20001)
    R = sol.sol(z)[0]
    return L, float(_trapezoid(R * R, z))


def closed_loop(u0, Cpmax, V0, ratio0=1.0, **kw):
    """Solve Cpn = Cpmax (ratio0 V0/V - 1) self-consistently.

    `ratio0` is the pressure ratio p_i(0)/p_e of the base state and must be
    passed whenever that state has Cpn != 0.  Omitting it, as an earlier
    version of this function forced one to do, closes the loop about
    Cpn = 0 instead of about the base state, so that at u0 = 1 the solve
    does not return the base value and the resulting gain is meaningless.
    That defect was invisible for the case reported in the paper, which has
    Cpn = 0, and only appeared when the same computation was repeated at
    other parameter sets.
    """
    def g(Cpn):
        _, V = march(u0, Cpn, **kw)
        return Cpn - Cpmax * (ratio0 * V0 / V - 1.0)
    Cpn = brentq(g, -0.9, 0.9, xtol=1.0e-13, rtol=1.0e-14)
    L, V = march(u0, Cpn, **kw)
    return L, V, Cpn


if __name__ == "__main__":
    Fr, We, Cpmax = 10.0, 50.0, 1.0
    L0, V0 = march(1.0, 0.0, Fr=Fr, We=We)
    print(f"Estado base:  L = {L0:.10f}   V = {V0:.10f}   C_pn = 0")
    print(f"Margen critico en la boquilla, We u0^2 - cos(th0) = {We - 1:.4f}")
    print()

    for d in (1.0e-3, 1.0e-4):
        Lp, _ = march(1.0 + d, 0.0, Fr=Fr, We=We)
        Lm, _ = march(1.0 - d, 0.0, Fr=Fr, We=We)
        g_open = (Lp - Lm) / (2.0 * d)
        Lpc, _, Cp_p = closed_loop(1.0 + d, Cpmax, V0, Fr=Fr, We=We)
        Lmc, _, Cp_m = closed_loop(1.0 - d, Cpmax, V0, Fr=Fr, We=We)
        g_closed = (Lpc - Lmc) / (2.0 * d)
        dCpn = (Cp_p - Cp_m) / (2.0 * d)
        print(f"delta = {d:.0e}")
        print(f"   dL/du0  lazo abierto (C_pn=0 fijo) = {g_open:12.6f}"
              f"   normalizada {g_open / L0:.6f}")
        print(f"   dL/du0  lazo cerrado              = {g_closed:12.6f}"
              f"   normalizada {g_closed / L0:.6f}")
        print(f"   dC_pn/du0                         = {dCpn:12.6f}")
