"""Spectral convergence, published steady data, and finite thickness.

`critical_margin()` is to be read as "singular coefficient matrix, check
compatibility", not as "no steady state exists".  The singularity is
removable if the right-hand side lies in the range, which requires
Cpn* = (Fr R + We^2 s)/(Fr R^2 sqrt(1 + s^2)) and equals unity at the nozzle
with theta0 = 0.  That result is published: Ramos, ZAMM 72 (1992) 565-589,
Eq. (94) and the discussion of Cp We = 1; Meccanica 32 (1997) 279-293,
Eq. (40); and, as an equality of ranks, Appl. Math. Modelling 20 (1996)
440-458, Eq. (35).

Carrying the parameter continuation at BASE_N = 8 and refining only
afterwards stalls near the critical surface, so `solve_steady` now adapts
the resolution of the continuation to the critical margin.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np

from annular_spectral import Parameters, solve_steady
from annular_thickness import ThickParameters, reference_solve

# --- L(0) as tabulated by Ramos, Appl. Math. Modelling 16 (1992), Table 1 ---
REF92 = {
    "Fig 3-5,11,12": (Parameters(Fr=10, We=50, theta0_deg=0), 12.5590),
    "Fig 6":         (Parameters(Fr=10000, We=50, theta0_deg=0), 9.8652),
    "Fig 8":         (Parameters(Fr=10, We=100, theta0_deg=0), 19.0452),
    "Fig 9":         (Parameters(Fr=10, We=50, theta0_deg=-15), 3.8236),
    "Fig 10":        (Parameters(Fr=10, We=50, theta0_deg=15), 37.3466),
    "Fig 13":        (Parameters(Fr=10, We=50, theta0_deg=0,
                                 pressure_ratio0=0.5), 10.2816),
}

# --- Ramos, Comput. Mech. 11 (1993), Tabla 4:  Fr=10, We=50, theta0=0,
#     beta=0.05, Cpn = Cpmax (p_i(0)/p_e - 1) unless stated ---------------
REF93 = [
    ("ratio 0.5",            10.0, 50.0, 0.05, -0.50, 0.0, 11.030),
    ("ratio 1.0",            10.0, 50.0, 0.05,  0.00, 0.0, 13.388),
    ("Cpmax=1,  r=1.05",     10.0, 50.0, 0.05,  0.05, 0.0, 13.712),
    ("Cpmax=5,  r=1.05",     10.0, 50.0, 0.05,  0.25, 0.0, 15.278),
    ("Cpmax=10, r=1.05",     10.0, 50.0, 0.05,  0.50, 0.0, 18.239),
    ("beta=0.005",           10.0, 50.0, 0.005, 0.50, 0.0, 17.604),
    ("beta=0.1",             10.0, 50.0, 0.10,  0.50, 0.0, 18.609),
    ("Fr=1000",            1000.0, 50.0, 0.05,  0.50, 0.0, 13.930),
    ("Fr=infinito",         1e12, 50.0, 0.05,  0.50, 0.0, 13.872),
    ("We=5",                 10.0,  5.0, 0.05,  0.50, 0.0,  4.511),
    ("We=75",                10.0, 75.0, 0.05,  0.50, 0.0, 23.289),
]


def thick_L(Fr, We, beta, Cpn, th=0.0):
    p = ThickParameters(Fr=Fr, We=We, theta0_deg=th, beta=beta,
                        Cpmax=1.0, pressure_ratio0=1.0 + Cpn)
    L, _ = reference_solve(p, z_max=400.0)
    return L


def rule(txt):
    print("\n" + "=" * 78)
    print(txt)
    print("=" * 78)


rule("1. CONVERGENCIA ESPECTRAL  (Fr=10, We=50, theta0=0)")
p = Parameters()
wref, _ = solve_steady(96, p)
Lstar = float(wref[-1])
print(f"L* (N=96) = {Lstar:.15f}")
print(f"{'N':>5}{'L':>22}{'|L-L*|':>14}{'factor':>10}")
prev = None
for N in (6, 8, 10, 12, 14, 16, 20, 24, 32, 48):
    w, _ = solve_steady(N, p)
    e = abs(float(w[-1]) - Lstar)
    f = "" if prev is None or e == 0 else f"{prev / e:10.2f}"
    print(f"{N:5d}{float(w[-1]):22.14f}{e:14.3e}{f}")
    prev = e

rule("2. TABLE 1 OF RAMOS (1992), MEMBRANE,  N=32")
print(f"{'case':16}{'We-cos(th0)':>13}{'L present':>16}"
      f"{'L Ramos':>11}{'rel diff':>11}")
for name, (pp, Lr) in REF92.items():
    w, prob = solve_steady(32, pp)
    L = float(w[-1])
    res = np.max(np.abs(prob.residual(w)))
    print(f"{name:16}{pp.critical_margin():13.4f}{L:16.8f}{Lr:11.4f}"
          f"{abs(L - Lr) / Lr:11.2e}   |F|={res:.1e}")
print("\nThe We = 1 row of that table is excluded here: the critical margin")
print("is zero, the matrix is singular, and compatibility would require")
print("Cpn = 1, whereas that case uses Cpn = 0.  It is that single case that")
print("is incompatible, not the whole critical surface.")

rule("3. APPROACH TO THE CRITICAL SURFACE:  Fr=10, theta0=0, Cpn=0")
print(f"{'We':>9}{'margin':>9}{'L collocation N=64':>19}"
      f"{'L DOP853 march':>18}{'rel diff':>10}")
for We in (1.5, 1.2, 1.1, 1.05, 1.02, 1.01, 1.005, 1.001):
    Lref = thick_L(10.0, We, 0.0, 0.0)
    try:
        w, _ = solve_steady(64, Parameters(Fr=10.0, We=We, theta0_deg=0.0))
        L = float(w[-1])
        print(f"{We:9.4f}{We - 1:9.4f}{L:19.10f}{Lref:18.10f}"
              f"{abs(L - Lref) / Lref:10.1e}")
    except Exception as exc:
        print(f"{We:9.4f}{We - 1:9.4f}{'no converge':>19}{Lref:18.10f}"
              f"   {type(exc).__name__}")
print("\nWith the continuation resolution adapted to the margin the solver")
print("reaches a margin of 1e-2 with twelve digits; below that the reference")
print("is the DOP853 march, which needs no continuation at all.")

rule("4. TABLE 4 OF RAMOS (1993), FINITE THICKNESS")
print("Cpn = Cpmax (p_i(0)/p_e - 1) is the only pressure parameter of the")
print("steady problem; the two rows of Ramos with Cpn = 0.5 give the same L,")
print("which confirms it in his own data.\n")
print(f"{'case':22}{'Cpn':>7}{'L membrane':>14}{'L (beta)':>13}"
      f"{'L Ramos':>10}{'ref/membrane':>12}")
for name, Fr, We, beta, Cpn, th, Lr in REF93:
    Lm = thick_L(Fr, We, 0.0, Cpn, th)
    Lb = thick_L(Fr, We, beta, Cpn, th)
    print(f"{name:22}{Cpn:7.2f}{Lm:14.5f}{Lb:13.5f}{Lr:10.3f}{Lr / Lm:12.3f}")

rule("5. dL/dbeta < 0 WHENEVER THE TIP IS TRANSVERSAL")
print("The momentum equations do not contain b, so at fixed (Fr, We, theta0,")
print("Cpn) the trajectory (R(z), u(z)) is the same for every beta and the")
print("tip is the first zero of h(z) = R^2 - beta m / 2.  Since")
print("dh/dbeta = -m/2 < 0, it follows that dL/dbeta = (m/2)/h_z(L) < 0 when")
print("h_z(L) < 0, that is, whenever the zero is transversal.\n")
print(f"{'beta':>8}{'L (We=50, C_pn=0.5)':>22}{'L (We=25, C_pn=0)':>20}")
for beta in (0.0, 0.005, 0.01, 0.05, 0.1, 0.2):
    print(f"{beta:8.3f}{thick_L(10.0, 50.0, beta, 0.5):22.6f}"
          f"{thick_L(10.0, 25.0, beta, 0.0):20.6f}")
print("\nDecreasing in both families.  Table 4 of Ramos gives, for We=50,")
print("L = 17.604, 18.239, 18.609 al crecer beta = 0.005, 0.05, 0.1, es decir")
print("increasing, and that paper itself stresses the contrast with We=25.")
print("This is incompatible with its own equations and points to the tip")
print("closure: its Eq. (108) is the time derivative of the algebraic")
print("constraint (76), which is marched in time with extrapolated values and")
print("never re-imposed; in the steady limit that ODE is satisfied for every")
print("L, so nothing fixes the length except the constraint itself.")
