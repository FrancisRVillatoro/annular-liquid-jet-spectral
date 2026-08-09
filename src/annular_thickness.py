"""
Thin annular liquid jets of finite thickness: a formulation that is uniform
in the thickness parameter beta = b_0* / R_0*.

Model: Ramos, Appl. Math. Modelling 16 (1992) 464-475; Int. J. Numer.
Methods Fluids 21 (1995) 735-761.  The momentum equations do not contain
b; the thickness enters only through

  (i)  the tip condition  R_i(t, L) = 0, with R_i = R - b/2 and
       b = beta m / R, equivalently  g = 2 R^2 - beta m = 0;
  (ii) the enclosed volume  V = L int_0^1 R_i^2 d(eta).

Since the determinant of the steady system is still D = We u - J, the whole
critical structure -- the critical surface, the regularity condition Cpn*,
and the saddle character of the critical point -- is independent of beta.
The driver of this module checks that.

UNIFORM FORMULATION.  The numerical obstacle is that g = 2 R^2 - beta m
defines an index-two constraint whose regularity is measured by

    Delta = 4 R R_eta - beta m_eta   at eta = 1,

which degenerates in the membrane limit beta -> 0 because R -> 0 there.
Instead the inner radius is taken as the unknown and the constraint is
absorbed:

    R_i(eta) = (1 - eta) Sigma(eta),
    R        = [ R_i + sqrt(R_i^2 + 2 beta m) ] / 2,

which inverts R_i = R - beta m / (2 R).  With this choice:

  - the tip constraint is satisfied IDENTICALLY, not to the tolerance of
    the integrator, so there is no index-two drift-off and neither
    projection nor GGL-type stabilisation is needed;
  - beta = 0 returns exactly R = R_i = (1 - eta) Sigma, the membrane
    substitution;
  - Sigma is analytic up to the tip, so convergence remains geometric;
  - the speed of the free boundary is

        dL/dt = u(1) + L [ v + (dR/dm)(m/L) u_eta ]
                        / [ (dR/dR_i) Sigma ]   at eta = 1,

    which reduces to dL/dt = u(1) + L v(1)/Sigma(1) at beta = 0 and is
    algebraically equivalent to Eq. (39) of Ramos (1992), here without any
    0/0 quotient evaluated numerically.

At the nozzle, R(0) = 1 and m(0) = 1 give Sigma(0) = R_i(0) = 1 - beta/2.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root

from annular_spectral import (
    Parameters,
    barycentric_interpolate,
    cgl_nodes_and_diff,
    clenshaw_curtis_weights,
)


# ----------------------------------------------------------------------
# Parameters with thickness
# ----------------------------------------------------------------------

@dataclass
class ThickParameters(Parameters):
    """The Parameters of annular_spectral plus the thickness beta = b_0*/R_0*."""

    beta: float = 0.0

    def nozzle_inner_radius(self) -> float:
        return 1.0 - 0.5 * self.beta


# ----------------------------------------------------------------------
# Algebra of the substitution
# ----------------------------------------------------------------------

def mean_radius(R_i: np.ndarray, m: np.ndarray, beta: float):
    """Return R, dR/dR_i and dR/dm from the inner radius and the mass.

    R solves R^2 - R_i R - beta m / 2 = 0, taking the positive root.
    """
    if beta == 0.0:
        # Membrane limit: R = R_i exactly.  The branch with W = |R_i| is
        # singular at the tip, so it is handled separately.
        return R_i, np.ones_like(R_i), np.zeros_like(R_i)

    W = np.sqrt(R_i * R_i + 2.0 * beta * m)
    R = 0.5 * (R_i + W)
    dR_dRi = 0.5 * (1.0 + R_i / W)
    dR_dm = 0.5 * beta / W
    return R, dR_dRi, dR_dm


# ----------------------------------------------------------------------
# Steady problem
# ----------------------------------------------------------------------

class ThickSteady:
    """Steady problem with finite thickness; unknowns (Sigma, u, L).

    It uses m = 1/u (from m u = 1) and the same equations (A), from energy,
    and (B), from radial momentum, as the membrane case, with R obtained
    from Sigma and u through the substitution.
    """

    def __init__(self, N: int, params: ThickParameters):
        self.N = N
        self.p = params
        self.eta, self.D = cgl_nodes_and_diff(N)
        self.om = 1.0 - self.eta
        self.wq = clenshaw_curtis_weights(N)

    def steady_cpn(self) -> float:
        return self.p.Cpmax * (self.p.pressure_ratio0 - 1.0)

    def unpack(self, w):
        N = self.N
        return w[: N + 1], w[N + 1 : 2 * N + 2], float(w[-1])

    def geometry(self, Sigma, u):
        """R, R_eta and R_etaeta from (Sigma, u) with m = 1/u."""
        beta = self.p.beta
        D = self.D

        m = 1.0 / u
        m_eta = -(D @ u) / (u * u)
        m_etaeta = (2.0 * (D @ u) ** 2 - u * (D @ (D @ u))) / u ** 3

        DS = D @ Sigma
        R_i = self.om * Sigma
        Ri_eta = -Sigma + self.om * DS
        Ri_etaeta = -2.0 * DS + self.om * (D @ DS)

        if beta == 0.0:
            return R_i, Ri_eta, Ri_etaeta, R_i

        W = np.sqrt(R_i * R_i + 2.0 * beta * m)
        W_eta = (R_i * Ri_eta + beta * m_eta) / W
        W_etaeta = (
            Ri_eta ** 2 + R_i * Ri_etaeta + beta * m_etaeta - W_eta ** 2
        ) / W

        R = 0.5 * (R_i + W)
        R_eta = 0.5 * (Ri_eta + W_eta)
        R_etaeta = 0.5 * (Ri_etaeta + W_etaeta)
        return R, R_eta, R_etaeta, R_i

    def residual(self, w):
        Sigma, u, L = self.unpack(w)
        if L <= 0.0 or np.min(u) <= 0.0:
            return np.full(w.size, 1.0e6)

        R, R_eta, R_etaeta, _ = self.geometry(Sigma, u)
        u_eta = self.D @ u

        s = R_eta / L
        sp = R_etaeta / (L * L)
        q = 1.0 + s * s
        Cpn = self.steady_cpn()

        A = (1.0 + s * s) * u_eta / L + s * u * sp - 1.0 / (self.p.Fr * u)
        B = (
            self.p.We * s * u_eta / L
            + (self.p.We * u - R * q ** -1.5) * sp
            - (Cpn * R - q ** -0.5)
        )

        return np.concatenate(
            [
                A[1:],
                B[1:],
                [
                    Sigma[0] - self.p.nozzle_inner_radius(),
                    u[0] - 1.0,
                    R_eta[0] - L * self.p.tan_theta0,
                ],
            ]
        )

    def determinant(self, w):
        """D = We u - J along the solution."""
        Sigma, u, L = self.unpack(w)
        R, R_eta, _, _ = self.geometry(Sigma, u)
        s = R_eta / L
        return self.p.We * u - R / np.sqrt(1.0 + s * s)

    def tip_residual(self, w) -> float:
        """g = 2 R^2 - beta m at the tip.  Zero by construction."""
        Sigma, u, L = self.unpack(w)
        R, _, _, _ = self.geometry(Sigma, u)
        return float(2.0 * R[-1] ** 2 - self.p.beta / u[-1])

    def volume(self, w) -> float:
        Sigma, u, L = self.unpack(w)
        R_i = self.om * Sigma
        return float(L * np.dot(self.wq, R_i * R_i))

    def solve(self, guess, tol: float = 1.0e-13):
        return root(
            self.residual,
            guess,
            method="hybr",
            tol=tol,
            options={"maxfev": 40000},
        )


def _converged(solution, tol: float = 1.0e-9) -> bool:
    """Accept a Newton solve on its residual, not on the solver flag.

    `hybr` reports success=False once it can no longer improve, even when the
    residual is already at round-off.
    """
    return bool(solution.success) or float(
        np.max(np.abs(solution.fun))
    ) < tol


BASE_N = 8


def _blend(start: ThickParameters, target: ThickParameters, lam: float):
    return ThickParameters(
        Fr=1.0 / ((1.0 - lam) / start.Fr + lam / target.Fr),
        We=(1.0 - lam) * start.We + lam * target.We,
        theta0_deg=(1.0 - lam) * start.theta0_deg + lam * target.theta0_deg,
        Cpmax=target.Cpmax,
        pressure_ratio0=(1.0 - lam) * start.pressure_ratio0
        + lam * target.pressure_ratio0,
        amplitude=0.0,
        St=target.St,
        # The continuation is linear in sqrt(beta), not in beta: the tip sits
        # at R = sqrt(beta m / 2), so that L(0) - L(beta) = O(sqrt(beta))
        # y dL/dbeta diverge en beta = 0.
        beta=(
            (1.0 - lam) * math.sqrt(start.beta) + lam * math.sqrt(target.beta)
        ) ** 2,
    )


def solve_steady_thick(N: int, target: ThickParameters, steps: int = 20):
    """Solve the steady problem at resolution N.

    The seed is built from the reference ODE march and then refined by
    Newton along a ladder in N.  There is no continuation in beta from
    zero: the tip sits at R = sqrt(beta m / 2), so the reconstruction
    R(R_i, m) has a layer of width O(sqrt(beta)) next to eta = 1 that no
    fixed mesh resolves as beta -> 0.  It is also the reason why
    L(0) - L(beta) = O(sqrt(beta)) and dL/dbeta diverges there.
    """
    prob = ThickSteady(N, target)
    sol = prob.solve(guess_from_reference(N, target))
    if _converged(sol):
        return sol.x, prob

    # The layer of width O(sqrt(beta)) next to the tip may be unresolved
    # at low resolution, so the ladder starts from a finer mesh.
    if N >= 128:
        raise RuntimeError(
            f"No converge a N={N}; beta={target.beta:g} puede ser demasiado "
            "too small for this resolution."
        )
    w_fine, prob_fine = solve_steady_thick(2 * N, target)
    S = barycentric_interpolate(prob_fine.eta, w_fine[: prob_fine.N + 1],
                                prob.eta)
    u = barycentric_interpolate(
        prob_fine.eta,
        w_fine[prob_fine.N + 1 : 2 * prob_fine.N + 2],
        prob.eta,
    )
    sol = prob.solve(np.concatenate([S, u, [w_fine[-1]]]))
    if not _converged(sol):
        raise RuntimeError(f"No converge a N={N}.")
    return sol.x, prob


def _solve_steady_thick_continuation(N, target, steps=20):
    """Continuation-based variant, kept for the membrane limit."""
    start = ThickParameters(Fr=target.Fr, We=50.0, theta0_deg=0.0, beta=0.0)

    prob = ThickSteady(BASE_N, start)
    guess = np.concatenate(
        [
            np.linspace(1.0, 1.6, BASE_N + 1),
            np.linspace(1.0, 1.6, BASE_N + 1),
            [max(0.5, target.Fr * 50.0 / 40.0)],
        ]
    )
    sol = prob.solve(guess)
    if not _converged(sol):
        raise RuntimeError("The base solve failed.")
    w = sol.x

    lam, inc = 0.0, 1.0 / steps
    while lam < 1.0 - 1.0e-12:
        trial = min(1.0, lam + inc)
        prob = ThickSteady(BASE_N, _blend(start, target, trial))
        sol = prob.solve(w)
        if _converged(sol):
            w, lam = sol.x, trial
            inc = min(1.0 / steps, 1.5 * inc)
        else:
            inc *= 0.5
            if inc < 1.0e-7:
                raise RuntimeError(
                    f"Continuacion detenida en lambda={lam:.6f}; "
                    f"margin We-cos(theta0)={target.critical_margin():.3e}"
                )

    prob = ThickSteady(BASE_N, target)
    for n in [k for k in (12, 16, 24, 32, 48, 64, 96, 128) if k < N] + [N]:
        if n == BASE_N:
            continue
        nxt = ThickSteady(n, target)
        S = barycentric_interpolate(prob.eta, w[: prob.N + 1], nxt.eta)
        u = barycentric_interpolate(
            prob.eta, w[prob.N + 1 : 2 * prob.N + 2], nxt.eta
        )
        sol = nxt.solve(np.concatenate([S, u, [w[-1]]]))
        if not _converged(sol):
            raise RuntimeError(f"The resolution ladder failed at N={n}.")
        w, prob = sol.x, nxt

    return w, prob


# ----------------------------------------------------------------------
# Independent reference: ODE march in the physical coordinate
# ----------------------------------------------------------------------

def reference_solve(p: ThickParameters, z_max: float = 200.0):
    """Integrate the steady system in z with the event 2 R^2 - beta m = 0.

    A reference independent of the spectral discretisation: DOP853 with
    rtol 1e-12 and the tip located by the event.  Returns (L, dense
    solution).
    """
    Fr, We, beta = p.Fr, p.We, p.beta
    Cpn = p.Cpmax * (p.pressure_ratio0 - 1.0)

    def f(z, y):
        R, u, s = y
        q = 1.0 + s * s
        M = np.array([[q, s * u], [We * s, We * u - R * q ** -1.5]])
        b = np.array([1.0 / (Fr * u), Cpn * R - 1.0 / math.sqrt(q)])
        return [s, *np.linalg.solve(M, b)]

    if beta == 0.0:
        def event(z, y):
            return y[0]
    else:
        # The tip is the first zero of g = 2 R^2 - beta m, but g is even in
        # R: once R has passed through the root and continued to negative
        # values, g becomes large and positive again.  If a single step
        # spans the crossing -- which happens for small beta, where the
        # level sqrt(beta m / 2) is tiny -- the sign change is invisible and
        # the event is missed.  The equivalent condition R = sqrt(beta m/2)
        # is monotone through the crossing and is used instead.
        def event(z, y):
            return y[0] - math.sqrt(0.5 * beta / y[1])

    event.terminal = True
    event.direction = -1

    sol = solve_ivp(
        f,
        (0.0, z_max),
        [1.0, 1.0, p.tan_theta0],
        method="DOP853",
        events=event,
        rtol=1.0e-12,
        atol=1.0e-14,
        max_step=0.25,
        dense_output=True,
    )
    if sol.t_events[0].size != 1:
        return float("nan"), None
    return float(sol.t_events[0][0]), sol


def reference_length(p: ThickParameters, z_max: float = 200.0) -> float:
    return reference_solve(p, z_max)[0]


def guess_from_reference(N: int, p: ThickParameters, filter_degree: int = 12):
    """Spectral seed built from the ODE march.

    It is filtered by Chebyshev least squares before being evaluated at the
    nodes: dividing by (1 - eta) near the tip amplifies the noise of the
    dense output, and spectral differentiation amplifies it again by N^2.
    """
    from numpy.polynomial.chebyshev import chebfit, chebval

    L, sol = reference_solve(p)
    if sol is None:
        raise RuntimeError("The reference ODE march did not reach the tip.")

    zz = np.linspace(0.0, 1.0, 800)
    R, u, _ = sol.sol(L * zz)
    m = 1.0 / u
    R_i = R - 0.5 * p.beta * m / R

    # Sigma = R_i/(1-eta); the last value by l'Hopital's rule.
    Sigma = np.empty_like(zz)
    Sigma[:-1] = R_i[:-1] / (1.0 - zz[:-1])
    Sigma[-1] = (R_i[-3] / (1.0 - zz[-3]) * 3.0
                 - R_i[-4] / (1.0 - zz[-4]) * 2.0)

    deg = min(filter_degree, N)
    cS = chebfit(2.0 * zz - 1.0, Sigma, deg)
    cU = chebfit(2.0 * zz - 1.0, u, deg)

    eta, _ = cgl_nodes_and_diff(N)
    return np.concatenate(
        [chebval(2.0 * eta - 1.0, cS), chebval(2.0 * eta - 1.0, cU), [L]]
    )


# ----------------------------------------------------------------------
# Unsteady operator
# ----------------------------------------------------------------------

@dataclass
class ThickUnsteady:
    """Mapped unsteady equations with finite thickness.

    State: y = [m(1..N), Sigma(1..N), u(1..N), v(1..N), L], of size 4N+1.
    The nozzle values are imposed algebraically: m(0) = 1,
    Sigma(0) = 1 - beta/2, u(0) = u_0(t), v(0) = u_0(t) tan(theta0).
    """

    N: int
    p: ThickParameters
    volume_reference: float = 0.0
    eta: np.ndarray = field(init=False)
    D: np.ndarray = field(init=False)
    om: np.ndarray = field(init=False)
    wq: np.ndarray = field(init=False)

    def __post_init__(self):
        self.eta, self.D = cgl_nodes_and_diff(self.N)
        self.om = 1.0 - self.eta
        self.wq = clenshaw_curtis_weights(self.N)

    @property
    def size(self) -> int:
        return 4 * self.N + 1

    def pack(self, m, Sigma, u, v, L):
        return np.concatenate([m[1:], Sigma[1:], u[1:], v[1:], [L]])

    def unpack(self, y, t):
        N = self.N
        u0 = self.p.nozzle_velocity(t)
        m = np.empty(N + 1)
        S = np.empty(N + 1)
        u = np.empty(N + 1)
        v = np.empty(N + 1)
        m[0] = 1.0
        S[0] = self.p.nozzle_inner_radius()
        u[0] = u0
        v[0] = u0 * self.p.tan_theta0
        m[1:] = y[0:N]
        S[1:] = y[N : 2 * N]
        u[1:] = y[2 * N : 3 * N]
        v[1:] = y[3 * N : 4 * N]
        return m, S, u, v, float(y[-1])

    def volume(self, S, L) -> float:
        R_i = self.om * S
        return float(L * np.dot(self.wq, R_i * R_i))

    def pressure_coefficient(self, S, L) -> float:
        vol = self.volume(S, L)
        if vol <= 0.0:
            raise FloatingPointError(f"Nonpositive enclosed volume {vol}.")
        return self.p.Cpmax * (
            self.p.pressure_ratio0 * self.volume_reference / vol - 1.0
        )

    def tip_residual(self, y, t) -> float:
        """g = 2 R^2 - beta m at eta = 1.  Zero by construction."""
        m, S, u, v, L = self.unpack(y, t)
        R, _, _ = mean_radius(self.om * S, m, self.p.beta)
        return float(2.0 * R[-1] ** 2 - self.p.beta * m[-1])

    def rhs(self, t, y, Cpn_fixed=None):
        beta = self.p.beta
        m, S, u, v, L = self.unpack(y, t)

        if L <= 0.0 or np.min(m) <= 0.0 or np.min(u) <= 0.0:
            raise FloatingPointError(
                f"Estado no fisico: L={L}, min m={m.min()}, min u={u.min()}"
            )
        if S[-1] <= 0.0:
            raise FloatingPointError(f"Degenerate tip: Sigma(1)={S[-1]}")

        Dm = self.D @ m
        Du = self.D @ u
        Dv = self.D @ v
        DS = self.D @ S

        R_i = self.om * S
        Ri_eta = -S + self.om * DS
        Ri_etaeta = -2.0 * DS + self.om * (self.D @ DS)

        R, dR_dRi, dR_dm = mean_radius(R_i, m, beta)

        if beta == 0.0:
            R_eta, R_etaeta = Ri_eta, Ri_etaeta
        else:
            W = np.sqrt(R_i * R_i + 2.0 * beta * m)
            W_eta = (R_i * Ri_eta + beta * Dm) / W
            W_etaeta = (
                Ri_eta ** 2 + R_i * Ri_etaeta + beta * (self.D @ Dm)
                - W_eta ** 2
            ) / W
            R_eta = 0.5 * (Ri_eta + W_eta)
            R_etaeta = 0.5 * (Ri_etaeta + W_etaeta)

        s = R_eta / L
        q = 1.0 + s * s
        rt = np.sqrt(q)
        J_eta = R_eta / rt - R * R_eta * R_etaeta / (L * L * q * rt)
        Jeta_over_Reta = 1.0 / rt - R * R_etaeta / (L * L * q * rt)

        Cpn = (
            float(Cpn_fixed)
            if Cpn_fixed is not None
            else self.pressure_coefficient(S, L)
        )

        # Speed of the free boundary, obtained by requiring that the
        # numerator of Sigma_t vanish at eta = 1, which is equivalent to
        # d/dt (2R^2 - beta m) = 0 alli.
        Ldot = u[-1] + L * (
            v[-1] + dR_dm[-1] * (m[-1] / L) * Du[-1]
        ) / (dR_dRi[-1] * S[-1])

        a = (u - self.eta * Ldot) / L

        dm = -a * Dm - (m / L) * Du
        du = (
            -a * Du
            + 1.0 / self.p.Fr
            + (J_eta - Cpn * R * R_eta) / (m * self.p.We * L)
        )
        dv = -a * Dv + (Cpn * R - Jeta_over_Reta) / (m * self.p.We)

        # Sigma_t = G/(1-eta) - ..., with G(1) = 0 by the choice of dL/dt.
        G = (v - a * R_eta - dR_dm * dm) / dR_dRi
        H = np.empty_like(G)
        H[:-1] = G[:-1] / self.om[:-1]
        H[-1] = -float((self.D @ G)[-1])
        dS = H

        return np.concatenate([dm[1:], dS[1:], du[1:], dv[1:], [Ldot]])

    def state_from_steady(self, w, steady: ThickSteady):
        if steady.N != self.N:
            S = barycentric_interpolate(
                steady.eta, w[: steady.N + 1], self.eta
            )
            u = barycentric_interpolate(
                steady.eta, w[steady.N + 1 : 2 * steady.N + 2], self.eta
            )
            L = float(w[-1])
        else:
            S, u, L = steady.unpack(w)

        m = 1.0 / u
        R_i = self.om * S
        Ri_eta = -S + self.om * (self.D @ S)
        m_eta = -(self.D @ u) / (u * u)
        if self.p.beta == 0.0:
            R_eta = Ri_eta
        else:
            Wv = np.sqrt(R_i * R_i + 2.0 * self.p.beta * m)
            R_eta = 0.5 * (Ri_eta + (R_i * Ri_eta + self.p.beta * m_eta) / Wv)
        v = u * R_eta / L
        return self.pack(m, S, u, v, L)

    def jacobian(self, y, Cpn_fixed=None, eps=1.0e-7):
        f0 = self.rhs(0.0, y, Cpn_fixed=Cpn_fixed)
        J = np.empty((f0.size, y.size))
        for k in range(y.size):
            h = eps * max(1.0, abs(y[k]))
            yp = y.copy()
            yp[k] += h
            J[:, k] = (self.rhs(0.0, yp, Cpn_fixed=Cpn_fixed) - f0) / h
        return J

    def integrate(self, y0, t_final, method="DOP853",
                  rtol=1.0e-10, atol=1.0e-12, t_eval=None):
        return solve_ivp(
            lambda t, y: self.rhs(t, y),
            (0.0, t_final),
            y0,
            method=method,
            rtol=rtol,
            atol=atol,
            t_eval=t_eval,
            dense_output=True,
        )
