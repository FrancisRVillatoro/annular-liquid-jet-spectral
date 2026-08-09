"""A lag defined modulo T is ambiguous; it is unwrapped by continuation in St."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np
from annular_spectral import Parameters, UnsteadySpectral, solve_steady
from verify_dynamic import harmonic

base = Parameters(Fr=10, We=50, theta0_deg=0)
w, prob = solve_steady(32, base)
print(f"{'St':>7}{'T':>8}{'phase L':>12}{'phase Cpn':>12}"
      f"{'lag L':>12}{'lag Cpn':>13}{'A_L/L':>10}{'A_Cpn':>10}")
for St in (0.005, 0.01, 0.02, 0.04, 0.07, 0.1, 0.15, 0.2):
    p = Parameters(Fr=10, We=50, theta0_deg=0, amplitude=0.02, St=St,
                   ramp_cycles=2.0)
    T = p.period
    op = UnsteadySpectral(32, p, dealias=True)
    y0 = op.state_from_steady(w, prob)
    op.volume_reference = op.volume(op.unpack(y0, 0.0)[1], float(y0[-1]))
    NC = max(6, int(np.ceil(60.0/T)) + 4)
    te = np.linspace(0.0, NC*T, 60*NC+1)
    s = op.integrate(y0, NC*T, method="DOP853", rtol=1e-11, atol=1e-13,
                     t_eval=te, max_step=0.05)
    o = [op.observables(t, s.y[:,k]) for k,t in enumerate(s.t)]
    tt=np.array([x["t"] for x in o]); LL=np.array([x["L"] for x in o])
    CC=np.array([x["Cpn"] for x in o]); UU=np.array([x["u0"] for x in o])
    m = tt >= (NC-3)*T
    _,_,pu,_ = harmonic(tt[m], UU[m], St)
    Lm,AL,pL,_ = harmonic(tt[m], LL[m], St)
    _,AC,pC,_ = harmonic(tt[m], CC[m], St)
    # phase in (-pi, pi]: phase of the output relative to the input
    dphL = (pL-pu+np.pi)%(2*np.pi)-np.pi
    dphC = (pC-pu+np.pi)%(2*np.pi)-np.pi
    print(f"{St:7.3f}{T:8.1f}{dphL:12.5f}{dphC:12.5f}"
          f"{-dphL/(2*np.pi)*T:12.4f}{-dphC/(2*np.pi)*T:13.4f}"
          f"{AL/Lm:10.5f}{AC:10.5f}")
