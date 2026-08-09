"""Audit A1b: effect of de-aliasing on the results that depend on V."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np
from scipy.optimize import root
from annular_spectral import Parameters, solve_steady, UnsteadySpectral

def setup(Cpmax, Cpn, N, dealias, A=0.0, Stg=0.1):
    p = Parameters(Fr=10., We=50., theta0_deg=0., Cpmax=Cpmax,
                   pressure_ratio0=1.+Cpn/Cpmax, amplitude=0., ramp_cycles=0.,
                   body_amplitude=A, St_g=Stg)
    w, prob = solve_steady(N, p); L0 = prob.unpack(w)[2]
    uns = UnsteadySpectral(N=N, p=p, dealias=dealias)
    y0 = uns.state_from_steady(w, prob)
    uns.volume_reference = uns.volume(np.concatenate([[1.], y0[N:2*N]]), L0)
    return uns, y0, L0

def lead(Cpmax, Cpn, N, dealias):
    uns, y0, _ = setup(Cpmax, Cpn, N, dealias)
    ye = root(lambda y: uns.rhs(0., y), y0, method="hybr", tol=1e-13,
              options={"maxfev":40000}).x
    lam = np.linalg.eigvals(uns.jacobian(ye))
    return lam[int(np.argmax(lam.real))]

print("Rightmost eigenvalue (Cpn=0.5), Clenshaw-Curtis vs Gauss-Legendre")
for Cpmax in (1.0, 3.4150592479, 10.0):
    for N in (24, 32):
        a = lead(Cpmax, 0.5, N, False); b = lead(Cpmax, 0.5, N, True)
        print(f"  Cpmax={Cpmax:<14.10g} N={N}: CC {a.real:+.10f}{a.imag:+.10f}i | "
              f"GL {b.real:+.10f}{b.imag:+.10f}i | |d| {abs(a-b):.2e}")

def tstar(N, dealias, eps=1e-6):
    uns, y0, _ = setup(1.0, 0.0, N, dealias, A=0.5, Stg=0.5)
    def ev(t,y): return float(uns.unpack(y,t)[1][-1]) - eps
    ev.terminal=True; ev.direction=-1
    s = uns.integrate(y0, 30.0, method="DOP853", rtol=1e-10, atol=1e-12,
                      max_step=0.02, events=ev)
    return float(s.t_events[0][0]) if s.t_events[0].size else None

print()
print("Degeneracy time (body force, A=0.5, St_g=0.5), Clenshaw-Curtis vs Gauss-Legendre")
for N in (24, 32, 48, 64):
    print(f"  N={N:3d}: CC {tstar(N,False):.9f}   GL {tstar(N,True):.9f}")
