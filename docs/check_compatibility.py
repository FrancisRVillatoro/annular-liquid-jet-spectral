"""Symbolic check that Eq. (13) of the paper is Eq. (35) of Ramos (1996).

Ramos, Appl. Math. Modelling 20, 440-458 (1996), Eq. (35), reads

    +- gamma / (2 F u) sin(2 theta) + Delta_p R + (2 / W) cos(theta) = 0,

where gamma is the volumetric flow rate at the nozzle, F the Froude number,
W the Weber number, Delta_p = p_e - p_i non-dimensionalised by inertia, and
theta the local slope angle.  The dictionary to the notation of the present
paper is

    gamma = 1                 (from m u = 1),
    F     = Fr                (fixed by that paper's own Torricelli law
                               u = (1 + 2 z / F)^(1/2)),
    W     = 2 We              (since its W0^2 = gamma W / 2 is our We),
    Delta_p = - Cpn / We      (opposite sign, different normalisation),
    tan(theta) = s = dR/dz.

Run:  python3 check_compatibility.py
"""

import sympy as sp

R, u, s, Fr, We, Cpn = sp.symbols("R u s Fr We Cpn", positive=True)
q = 1 + s**2

# --- compatibility condition of the present paper -----------------------
M = sp.Matrix([[q, s * u],
               [We * s, We * u - R * q**sp.Rational(-3, 2)]])
b = sp.Matrix([1 / (Fr * u), Cpn * R - q**sp.Rational(-1, 2)])

det = sp.simplify(M.det() - (We * u - R / sp.sqrt(q)))
assert det == 0, "the determinant is not We*u - J"

lam = We * s / q                                   # row2 = lam * row1
u_crit = sp.solve(sp.Eq(We * u, R / sp.sqrt(q)), u)[0]
ours = sp.simplify(sp.solve(sp.Eq(b[1], lam * b[0]), Cpn)[0].subs(u, u_crit))

target = (Fr * R + We**2 * s) / (Fr * R**2 * sp.sqrt(q))
assert sp.simplify(ours - target) == 0, "Eq. (13) not recovered"

# --- Eq. (35) of Ramos (1996), translated -------------------------------
gamma, F, W, Dp = 1, Fr, 2 * We, -Cpn / We
eq35 = (gamma / (2 * F * u) * (2 * s / q)          # sin(2 theta) = 2 s / q
        + Dp * R
        + (2 / W) * 1 / sp.sqrt(q))                # cos(theta) = q^(-1/2)
theirs = sp.simplify(sp.solve(sp.Eq(eq35.subs(u, u_crit), 0), Cpn)[0])

assert sp.simplify(sp.together(theirs - ours)) == 0, "the two differ"

print("Eq. (13) of the paper :", sp.simplify(ours))
print("Eq. (35) of Ramos 1996:", sp.simplify(theirs))
print("equivalent            : True")
