"""Audit A1: aliasing of the quadrature for the enclosed volume."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np
from annular_spectral import Parameters, solve_steady, UnsteadySpectral

def audit(N, Cpn=0.5, Cpmax=1.0, Fr=10.0, We=50.0):
    p = Parameters(Fr=Fr, We=We, theta0_deg=0.0, Cpmax=Cpmax,
                   pressure_ratio0=1.0 + Cpn/Cpmax, amplitude=0.0, ramp_cycles=0.0)
    w, prob = solve_steady(N, p)
    S, u, L = prob.unpack(w)
    cc = UnsteadySpectral(N=N, p=p, dealias=False)
    gl = UnsteadySpectral(N=N, p=p, dealias=True)
    Vcc, Vgl = cc.volume(S, L), gl.volume(S, L)
    return L, Vcc, Vgl

print("Enclosed volume V = L * int_0^1 R^2 deta, steady state Fr=10 We=50 Cpn=0.5")
print(f"{'N':>4s} {'V (Clenshaw-Curtis)':>22s} {'V (Gauss-Legendre)':>21s} {'|dV|/V':>10s}")
Vref = None
for N in (8, 12, 16, 24, 32, 48, 64):
    L, Vcc, Vgl = audit(N)
    if Vref is None: Vref = Vgl
    print(f"{N:4d} {Vcc:22.15f} {Vgl:21.15f} {abs(Vcc-Vgl)/Vgl:10.2e}")
print()
print("Convergencia en N de V (Gauss-Legendre), referencia N=64")
_,_,V64 = audit(64)
for N in (8, 12, 16, 24, 32, 48):
    _,_,V = audit(N)
    print(f"  N={N:3d}  |V-V64| = {abs(V-V64):.3e}")
