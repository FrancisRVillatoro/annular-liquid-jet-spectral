"""
Critical structure of the steady problem for annular liquid membranes.

The steady system in the physical coordinate z, with unknowns (R, u,
s = R_z) and m = 1/u (from m u = 1), is

    [ 1+s^2      s u                  ] [u_z]   [ 1/(Fr u)                ]
    [ We s     We u - R (1+s^2)^{-3/2}] [s_z] = [ Cpn R - (1+s^2)^{-1/2}  ]

whose determinant simplifies exactly to

    D = We u - J,        J = R / sqrt(1 + R_z^2).

D = 0 defines the critical surface.  For theta = 0 the condition is that the
local Weber number of the sheet, We u / R, equal unity, that is, that the
axial speed equal the Taylor-Culick speed sqrt(2 sigma / rho b*): the steady
system changes type on crossing it.

At a critical point a smooth solution exists only if the right-hand side
lies in the range of the matrix, which gives the regularity condition, of
de Laval nozzle type,

    Cpn* = (Fr R + We^2 s) / (Fr R^2 sqrt(1 + s^2)),

which at the nozzle (R = 1, s = tan theta0, and criticality We = cos theta0)
reduces to

    Cpn* = cos(theta0) + cos^2(theta0) sin(theta0) / Fr,

and equals exactly 1 for theta0 = 0, the value for which the membrane is
cylindrical.

Three regimes:
  We > cos theta0   supercritical at the nozzle; pure initial-value problem;
                    Cpn free.
  We = cos theta0   critical at the nozzle; Cpn is forced; the solution is
                    the unstable manifold of a saddle of the desingularised
                    field, with eigenvalues {+lambda, 0, -lambda}.
  We < cos theta0   subcritical; an interior throat exists and Cpn is an
                    eigenvalue.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.integrate import solve_ivp


# ----------------------------------------------------------------------
# Campos basicos
# ----------------------------------------------------------------------

def matrices(R: float, u: float, s: float, We: float, Fr: float, Cpn: float):
    q = 1.0 + s * s
    M = np.array([[q, s * u], [We * s, We * u - R * q ** -1.5]])
    b = np.array([1.0 / (Fr * u), Cpn * R - 1.0 / math.sqrt(q)])
    return M, b


def determinants(R, u, s, We, Fr, Cpn):
    """Return (D, N_u, N_s) with u_z = N_u/D and s_z = N_s/D."""
    M, b = matrices(R, u, s, We, Fr, Cpn)
    D = float(np.linalg.det(M))
    M1 = M.copy()
    M1[:, 0] = b
    M2 = M.copy()
    M2[:, 1] = b
    return D, float(np.linalg.det(M1)), float(np.linalg.det(M2))


def critical_determinant(R, u, s, We):
    """D = We u - J.  Negativo subcritico, positivo supercritico."""
    return We * u - R / math.sqrt(1.0 + s * s)


def Cpn_regularity(R: float, s: float, We: float, Fr: float) -> float:
    """Regularity condition at a generic critical point."""
    return (Fr * R + We * We * s) / (Fr * R * R * math.sqrt(1.0 + s * s))


def Cpn_regularity_nozzle(theta0_deg: float, Fr: float) -> float:
    """The particular case at the nozzle, where also We = cos(theta0)."""
    t = math.radians(theta0_deg)
    return math.cos(t) + math.cos(t) ** 2 * math.sin(t) / Fr


def critical_weber(theta0_deg: float) -> float:
    return math.cos(math.radians(theta0_deg))


# ----------------------------------------------------------------------
# Desingularised field and classification of the critical point
# ----------------------------------------------------------------------

def desingularised(sigma, y, We, Fr, Cpn):
    """dR/dsigma = s D, du/dsigma = N_u, ds/dsigma = N_s, dz/dsigma = D."""
    R, u, s = float(y[0]), float(y[1]), float(y[2])
    D, Nu, Ns = determinants(R, u, s, We, Fr, Cpn)
    return [s * D, Nu, Ns, D]


def desingularised_jacobian(y, We, Fr, Cpn, h=1.0e-7):
    J = np.zeros((3, 3))
    f0 = np.array(desingularised(0.0, y, We, Fr, Cpn)[:3])
    for k in range(3):
        yp = np.array(y, dtype=float)
        yp[k] += h
        J[:, k] = (np.array(desingularised(0.0, yp, We, Fr, Cpn)[:3]) - f0) / h
    return J


def classify_nozzle_critical_point(theta0_deg: float, Fr: float):
    """Eigenvalues of the desingularised field at the nozzle critical point.

    The result is {+lambda, 0, -lambda}: a saddle.  The unstable manifold is
    one-dimensional, so exactly two trajectories pass through the critical
    point, as in a de Laval nozzle.
    """
    We = critical_weber(theta0_deg)
    Cpn = Cpn_regularity_nozzle(theta0_deg, Fr)
    y0 = np.array([1.0, 1.0, math.tan(math.radians(theta0_deg))])
    D, Nu, Ns = determinants(*y0, We, Fr, Cpn)
    values, vectors = np.linalg.eig(desingularised_jacobian(y0, We, Fr, Cpn))
    order = np.argsort(-values.real)
    return {
        "We_c": We,
        "Cpn_star": Cpn,
        "residuals": (D, Nu, Ns),
        "eigenvalues": values[order].real,
        "unstable_direction": np.real(vectors[:, order[0]])
        / np.linalg.norm(np.real(vectors[:, order[0]])),
    }


def transcritical_branch(theta0_deg, Fr, sign=+1, sigma_max=400.0, seed=1.0e-8):
    """Integrate the unstable manifold of the nozzle critical point.

    Returns the solve_ivp object with state [R, u, s, z] against sigma.
    """
    info = classify_nozzle_critical_point(theta0_deg, Fr)
    We, Cpn = info["We_c"], info["Cpn_star"]
    y0 = np.array([1.0, 1.0, math.tan(math.radians(theta0_deg))])
    y0 = y0 + sign * seed * info["unstable_direction"]

    def tip(sigma, y, *args):
        return y[0]

    tip.terminal = True
    tip.direction = -1

    return solve_ivp(
        desingularised,
        (0.0, sigma_max),
        list(y0) + [0.0],
        args=(We, Fr, Cpn),
        method="DOP853",
        rtol=1.0e-12,
        atol=1.0e-14,
        events=tip,
        dense_output=True,
    ), We, Cpn


# ----------------------------------------------------------------------
# Downstream march in the physical coordinate
# ----------------------------------------------------------------------

def march(We, Fr, Cpn, theta0_deg=0.0, z_max=40.0,
          rtol=1.0e-10, atol=1.0e-12):
    """Integrate the steady system from the nozzle.

    If the nozzle is subcritical and Cpn is not the eigenvalue, the
    integration fails at the throat with D -> 0 and N_s -> infinity.
    """
    def f(z, y):
        M, b = matrices(y[0], y[1], y[2], We, Fr, Cpn)
        uz, sz = np.linalg.solve(M, b)
        return [y[2], uz, sz]

    def tip(z, y):
        return y[0]

    tip.terminal = True
    tip.direction = -1

    return solve_ivp(
        f,
        (0.0, z_max),
        [1.0, 1.0, math.tan(math.radians(theta0_deg))],
        method="DOP853",
        rtol=rtol,
        atol=atol,
        events=tip,
        dense_output=True,
    )


# ----------------------------------------------------------------------
# Verification driver
# ----------------------------------------------------------------------

if __name__ == "__main__":
    Fr = 10.0

    print("=" * 76)
    print("NOZZLE CRITICAL POINT: We = cos(theta0), Cpn forced")
    print("=" * 76)
    print(f"{'theta0':>8}{'We_c':>11}{'C_pn*':>12}{'max|D,Nu,Ns|':>15}"
          f"   eigenvalues")
    for th in (0.0, 5.0, 15.0, -15.0, 30.0, -30.0):
        info = classify_nozzle_critical_point(th, Fr)
        res = max(abs(v) for v in info["residuals"])
        ev = "  ".join(f"{v:+.5f}" for v in info["eigenvalues"])
        print(f"{th:8.1f}{info['We_c']:11.6f}{info['Cpn_star']:12.6f}"
              f"{res:15.2e}   {ev}")
    print("\nStructure {+lambda, 0, -lambda}: a saddle.  Two trajectories pass")
    print("through the critical point, as in a de Laval nozzle.")

    print()
    print("=" * 76)
    print("TRANSCRITICAL BRANCH (downstream unstable manifold)")
    print("=" * 76)
    for th in (0.0, 15.0, -15.0):
        for sign in (+1, -1):
            sol, We, Cpn = transcritical_branch(th, Fr, sign)
            R, u, s, z = sol.y
            closes = sol.t_events[0].size == 1
            if z[-1] <= 0.0:
                continue                      # upstream branch
            L = float(sol.y_events[0][0][3]) if closes else float("inf")
            flat = float(np.max(np.abs(R - 1.0)))
            print(f"theta0={th:6.1f}  We_c={We:.6f}  C_pn*={Cpn:.6f}   "
                  f"L={L:12.6f}   max|R-1|={flat:.2e}")
    print("\nFor theta0 = 0 the transcritical branch is the cylindrical membrane")
    print("(R identically 1, L infinite); this is the classical result Cpn = 1,")
    print("explained here as a regularity condition at the critical point.")

    print()
    print("=" * 76)
    print("SUBCRITICAL NOZZLE (We < cos theta0): Cpn is an eigenvalue")
    print("=" * 76)
    We = 0.5
    print(f"We={We}, theta0=0, D(0)={We-1.0:+.2f}")
    print(f"{'C_pn':>7}{'z end':>11}{'D end':>13}{'N_s end':>13}"
          f"{'s end':>14}  state")
    for Cpn in (0.0, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0, 1.01, 1.1, 1.5):
        sol = march(We, Fr, Cpn)
        R, u, s = sol.y[:, -1]
        D, _, Ns = determinants(R, u, s, We, Fr, Cpn)
        state = "smooth" if sol.status == 0 else "THROAT"
        print(f"{Cpn:7.2f}{sol.t[-1]:11.4f}{D:13.3e}{Ns:13.3e}"
              f"{s:14.3e}  {state}")
    print("\nOnly Cpn = 1 crosses the throat; every other value fails with")
    print("D -> 0 and N_s bounded away from zero, that is s_z = N_s/D -> infinity.")
