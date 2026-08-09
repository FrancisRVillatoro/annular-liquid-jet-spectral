"""
Spectral (Chebyshev collocation) solver for the free-boundary problem of
inviscid, isothermal annular liquid membranes.

Model: Ramos, Appl. Math. Modelling 16 (1992) 464-475, membrane limit
b0* = 0, nondimensional equations (16)-(20), (25)-(26), (30).

Key structural facts exploited here (see verify_* drivers):

1.  The tip eta = 1 is a transversal zero of R with R_z(L) finite and
    nonzero.  The substitution

        R(eta) = (1 - eta) S(eta)

    yields an S that is analytic on the closed interval, so Chebyshev
    collocation converges geometrically.  The constraint R(1) = 0 is
    absorbed exactly and never imposed by extrapolation.

2.  The steady system changes type where

        det = We * u - J = 0,          J = R / sqrt(1 + R_z^2),

    which at the nozzle reduces to We = cos(theta0).  On that surface the
    coefficient matrix is singular but the singularity is removable when
    the right-hand side lies in its range; the compatibility condition is

        Cpn* = (Fr R + We^2 s) / (Fr R^2 sqrt(1 + s^2)),

    which at the nozzle gives cos(theta0) + cos^2(theta0) sin(theta0)/Fr,
    and equals 1 for theta0 = 0 (the cylindrical membrane).  Criticality
    therefore means "singular coefficient matrix, check compatibility",
    not "no steady state exists".  For the underlying theory see Ramos,
    ZAMM 72 (1992) 565-589, Eq. (94) and the discussion of C_p We = 1,
    and Ramos, Meccanica 32 (1997) 279-293, Eq. (40).

3.  The convergence-length equation is the analytic limit

        Ldot = u(1) + L v(1) / S(1),

    with no numerically evaluated 0/0.

Conventions: eta in [0, 1] increasing, nozzle at eta = 0, tip at eta = 1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root


# ----------------------------------------------------------------------
# Chebyshev machinery on [0, 1]
# ----------------------------------------------------------------------

def cgl_nodes_and_diff(N: int) -> tuple[np.ndarray, np.ndarray]:
    """Chebyshev-Gauss-Lobatto nodes on [0,1], increasing, and d/d(eta).

    Returns eta with eta[0] = 0 (nozzle) and eta[N] = 1 (tip).
    """
    if N < 4:
        raise ValueError("Use N >= 4.")

    k = np.arange(N + 1)
    x = np.cos(np.pi * k / N)                      # 1 -> -1

    c = np.ones(N + 1)
    c[0] = c[N] = 2.0
    c = c * (-1.0) ** k

    X = np.tile(x, (N + 1, 1)).T
    dX = X - X.T

    D = np.outer(c, 1.0 / c) / (dX + np.eye(N + 1))
    D -= np.diag(D.sum(axis=1))

    eta = 0.5 * (1.0 - x)                          # 0 -> 1
    return eta, -2.0 * D                           # chain rule d/deta


def clenshaw_curtis_weights(N: int) -> np.ndarray:
    """Clenshaw-Curtis quadrature weights on the CGL nodes of [0,1].

    Ordered to match cgl_nodes_and_diff.  Exact for polynomials of
    degree <= N and spectrally accurate for analytic integrands, which
    matters because the enclosed volume feeds back into the pressure.
    """
    if N < 2:
        raise ValueError("Use N >= 2.")

    # Weights on [-1,1] for nodes cos(k pi / N), then halved for [0,1].
    theta = np.pi * np.arange(N + 1) / N
    w = np.zeros(N + 1)
    v = np.ones(N - 1)

    if N % 2 == 0:
        w[0] = w[N] = 1.0 / (N * N - 1.0)
        for k in range(1, N // 2):
            v -= 2.0 * np.cos(2.0 * k * theta[1:N]) / (4.0 * k * k - 1.0)
        v -= np.cos(N * theta[1:N]) / (N * N - 1.0)
    else:
        w[0] = w[N] = 1.0 / (N * N)
        for k in range(1, (N + 1) // 2):
            v -= 2.0 * np.cos(2.0 * k * theta[1:N]) / (4.0 * k * k - 1.0)

    w[1:N] = 2.0 * v / N
    return 0.5 * w[::-1]


def barycentric_interpolate(
    eta_src: np.ndarray,
    values: np.ndarray,
    eta_dst: np.ndarray,
) -> np.ndarray:
    """Barycentric interpolation from one CGL grid to another."""
    n = eta_src.size - 1
    w = np.ones(n + 1)
    w[0] = w[n] = 0.5
    w = w * (-1.0) ** np.arange(n + 1)

    out = np.empty(eta_dst.size)
    for i, x in enumerate(eta_dst):
        d = x - eta_src
        j = int(np.argmin(np.abs(d)))
        if abs(d[j]) < 1.0e-14:
            out[i] = values[j]
        else:
            out[i] = float(np.sum(w * values / d) / np.sum(w / d))
    return out


# ----------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------

def smooth_step(x: float) -> float:
    """C-infinity transition from 0 to 1 on [0, 1]."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    a = math.exp(-1.0 / x)
    b = math.exp(-1.0 / (1.0 - x))
    return a / (a + b)


@dataclass
class Parameters:
    """Nondimensional parameters of Ramos (1992), Table 1.

    `ramp_cycles` switches the forcing on through a C-infinity ramp.  With
    the abrupt sinusoidal start of the original paper the initial data and
    the nozzle boundary condition are only C-0 compatible: the steady
    field gives u_tau(0,0) = 0 while the boundary requires
    u_tau(0,0) = 2 pi St a.  The resulting weak discontinuity travels down
    the characteristic and a spectral discretisation rings on it.  A
    first-order upwind scheme merely smears it, which is why the original
    computations do not show the problem.  Set ramp_cycles = 0 to
    reproduce the incompatible forcing.
    """

    Fr: float = 10.0
    We: float = 50.0
    theta0_deg: float = 0.0
    Cpmax: float = 1.0
    pressure_ratio0: float = 1.0
    amplitude: float = 0.0
    St: float = 0.1
    ramp_cycles: float = 2.0
    # Body-force forcing of Ramos, Arch. Appl. Mech. 65 (1995), Eq. (9):
    # 1/Fr = 1/Fr0 * [1 + A sin(2 pi St_g t)], with a steady nozzle
    # boundary condition.  The initial steady field is then an exact
    # equilibrium at t = 0 and the forcing is C-infinity in time, so no
    # ramp is needed and no corner incompatibility arises.
    body_amplitude: float = 0.0
    St_g: float = 0.1

    @property
    def tan_theta0(self) -> float:
        return math.tan(math.radians(self.theta0_deg))

    @property
    def cos_theta0(self) -> float:
        return math.cos(math.radians(self.theta0_deg))

    @property
    def period(self) -> float:
        return 1.0 / self.St

    def nozzle_velocity(self, t: float) -> float:
        envelope = (
            1.0
            if self.ramp_cycles <= 0.0
            else smooth_step(t / (self.ramp_cycles * self.period))
        )
        return 1.0 + self.amplitude * envelope * math.sin(
            2.0 * math.pi * self.St * t
        )

    @property
    def protocol(self) -> str:
        """Name of the start-up protocol.

        C2.  Two runs with the same (a, St) but different ramp_cycles are
        *different problems* in the transient, and any time-resolved
        quantity -- in particular the time of a finite-time degeneracy --
        must be reported together with the protocol.

        'historical_transient' (ramp_cycles = 0) reproduces the abrupt
        switch-on of Ramos (1992): the nozzle data are incompatible with
        the initial field at the corner (t, eta) = (0, 0), a weak wave is
        launched along the characteristic, and there is no spectral
        convergence in the transient (the error stagnates at 1e-3..1e-2 and
        improves only algebraically).  Refining beyond N ~ 64 aborts.

        'periodic_orbit' (ramp_cycles > 0) reaches the same asymptotic
        limit cycle, with the transient error falling geometrically.

        Body-force forcing (body_amplitude != 0) needs no ramp: the initial
        field is an exact equilibrium and the forcing is C-infinity in time.
        The corner incompatibility is then of second order, not first --
        see the residual profile in `offgrid.py`.
        """
        if self.body_amplitude != 0.0:
            return "body_force"
        return ("historical_transient" if self.ramp_cycles == 0.0
                else "periodic_orbit")

    def inv_Fr(self, t: float) -> float:
        """Instantaneous 1/Fr; constant unless body_amplitude != 0."""
        return (1.0 / self.Fr) * (
            1.0
            + self.body_amplitude
            * math.sin(2.0 * math.pi * self.St_g * t)
        )

    def critical_margin(self) -> float:
        """det at the nozzle: We - cos(theta0).  Zero => no steady state."""
        return self.We - self.cos_theta0


# ----------------------------------------------------------------------
# Steady problem: unknowns (S, u, L), equations (A) and (B)
# ----------------------------------------------------------------------

class SteadySpectral:
    """Steady membrane in mapped coordinates, R = (1-eta) S.

    Unknowns: S(eta_0..eta_N), u(eta_0..eta_N), L.
    Equations: the energy combination (A) and the radial momentum (B),
    collocated at eta_1..eta_N, plus S(0)=1, u(0)=1, R_eta(0)=L tan(theta0).
    """

    def __init__(self, N: int, params: Parameters):
        self.N = N
        self.p = params
        self.eta, self.D = cgl_nodes_and_diff(N)
        self.om = 1.0 - self.eta

    def steady_cpn(self) -> float:
        return self.p.Cpmax * (self.p.pressure_ratio0 - 1.0)

    def unpack(self, w: np.ndarray):
        N = self.N
        return w[: N + 1], w[N + 1 : 2 * N + 2], float(w[-1])

    def residual(self, w: np.ndarray) -> np.ndarray:
        S, u, L = self.unpack(w)
        if L <= 0.0:
            return np.full(w.size, 1.0e6)

        DS = self.D @ S
        R = self.om * S
        R_eta = -S + self.om * DS
        R_etaeta = -2.0 * DS + self.om * (self.D @ DS)
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
                [S[0] - 1.0, u[0] - 1.0, R_eta[0] - L * self.p.tan_theta0],
            ]
        )

    def determinant(self, w: np.ndarray) -> np.ndarray:
        """det = We*u - J along the solution; sign change => type change."""
        S, u, L = self.unpack(w)
        DS = self.D @ S
        R = self.om * S
        s = (-S + self.om * DS) / L
        return self.p.We * u - R / np.sqrt(1.0 + s * s)

    def solve(
        self,
        guess: np.ndarray | None = None,
        tol: float = 1.0e-13,
    ):
        if guess is None:
            guess = self.default_guess()
        sol = root(
            self.residual,
            guess,
            method="hybr",
            tol=tol,
            options={"maxfev": 40000},
        )
        return sol

    def default_guess(self) -> np.ndarray:
        """Crude guess: flat S and u, L from a slender-jet estimate."""
        N = self.N
        L0 = max(0.5, self.p.We * self.p.Fr / 40.0)
        return np.concatenate(
            [
                np.linspace(1.0, 1.6, N + 1),
                np.linspace(1.0, 1.6, N + 1),
                [L0],
            ]
        )

    def primitives(self, w: np.ndarray):
        """Return m, R, u, v, L in the original variables."""
        S, u, L = self.unpack(w)
        R = self.om * S
        R_eta = -S + self.om * (self.D @ S)
        v = u * R_eta / L
        m = 1.0 / u
        return m, R, u, v, L


def _converged(solution, tol: float = 1.0e-9) -> bool:
    """Accept a Newton solve on its residual, not on the solver flag.

    `hybr` reports success=False once it can no longer improve, even when the
    residual is already at round-off, so the flag alone would reject perfectly
    converged solutions.
    """
    return bool(solution.success) or float(
        np.max(np.abs(solution.fun))
    ) < tol


BASE_PARAMETERS = Parameters(Fr=10.0, We=50.0, theta0_deg=0.0)
BASE_N = 8


def _blend(start: Parameters, target: Parameters, lam: float) -> Parameters:
    return Parameters(
        Fr=1.0 / ((1.0 - lam) / start.Fr + lam / target.Fr),
        We=(1.0 - lam) * start.We + lam * target.We,
        theta0_deg=(1.0 - lam) * start.theta0_deg + lam * target.theta0_deg,
        Cpmax=target.Cpmax,
        pressure_ratio0=(1.0 - lam) * start.pressure_ratio0
        + lam * target.pressure_ratio0,
        amplitude=0.0,
        St=target.St,
    )


def _parameter_ladder(
    N: int,
    target: Parameters,
    guess: np.ndarray,
    steps: int,
) -> np.ndarray:
    """Walk from BASE_PARAMETERS to `target` at fixed N, halving on stall."""
    w = guess
    lam = 0.0
    increment = 1.0 / steps

    while lam < 1.0 - 1.0e-12:
        trial = min(1.0, lam + increment)
        prob = SteadySpectral(N, _blend(BASE_PARAMETERS, target, trial))
        sol = prob.solve(guess=w)
        if _converged(sol):
            w, lam = sol.x, trial
            increment = min(1.0 / steps, 1.5 * increment)
        else:
            increment *= 0.5
            if increment < 1.0e-6:
                raise RuntimeError(
                    f"Parameter continuation stalled at lambda={lam:.6f} "
                    f"(N={N}); margin We-cos(theta0)="
                    f"{target.critical_margin():.3e}"
                )
    return w


def solve_steady(
    N: int,
    target: Parameters,
    steps: int = 16,
    base_N: int | None = None,
) -> tuple[np.ndarray, SteadySpectral]:
    """Steady solve by parameter continuation, then a ladder in N.

    A flat initial guess only converges near the base parameter set and at
    low resolution, so both ladders are needed; this is a property of the
    problem, not of the discretisation.

    Correction C7.  Doing the parameter continuation at BASE_N = 8 and only
    then refining fails for We - cos(theta0) <~ 0.02: near criticality the
    solution develops a short scale next to the nozzle of width O(margin)
    that N = 8 cannot resolve, and the continuation stalls.  `base_N` now
    adapts to the margin, so the continuation is carried out at a resolution
    that already resolves that scale.  Pass base_N explicitly to override.
    """
    if base_N is None:
        margin = abs(target.critical_margin())
        if margin >= 0.5:
            base_N = BASE_N
        elif margin >= 0.1:
            base_N = 24
        elif margin >= 0.02:
            base_N = 48
        else:
            base_N = 96
        base_N = min(base_N, max(N, BASE_N))

    base = SteadySpectral(base_N, BASE_PARAMETERS)
    sol = base.solve()
    if not _converged(sol):
        raise RuntimeError(f"Base solve failed: {sol.message}")

    w = _parameter_ladder(base_N, target, sol.x, steps)
    prob = SteadySpectral(base_N, target)

    ladder = [n for n in (12, 16, 24, 32, 48, 64, 96, 128, 192)
              if base_N < n < N]
    for n in ladder + [N]:
        if n == base_N:
            continue
        nxt = SteadySpectral(n, target)
        S = barycentric_interpolate(prob.eta, w[: prob.N + 1], nxt.eta)
        u = barycentric_interpolate(
            prob.eta, w[prob.N + 1 : 2 * prob.N + 2], nxt.eta
        )
        sol = nxt.solve(guess=np.concatenate([S, u, [w[-1]]]))
        if not _converged(sol):
            raise RuntimeError(f"Resolution ladder failed at N={n}.")
        w, prob = sol.x, nxt

    return w, prob


# Backwards-compatible alias.
steady_continuation = solve_steady


# ----------------------------------------------------------------------
# Unsteady problem
# ----------------------------------------------------------------------

@dataclass
class UnsteadySpectral:
    """Mapped unsteady membrane equations, spectral in eta.

    State: y = [m(1..N), S(1..N), u(1..N), v(1..N), L], size 4N+1.
    Nozzle values are algebraic: m(0)=1, S(0)=1, u(0)=u0(t),
    v(0)=u0(t) tan(theta0).  All four characteristics enter at eta=0 and
    none at eta=1, so no outflow condition is needed or imposed.
    """

    N: int
    p: Parameters
    volume_reference: float = 0.0
    dealias: bool = False
    guard: str = "clamp"          # "clamp" or "raise"; see rhs
    floor: float = 1.0e-12
    n_clamped: int = 0
    eta: np.ndarray = field(init=False)
    D: np.ndarray = field(init=False)
    om: np.ndarray = field(init=False)
    wq: np.ndarray = field(init=False)
    P_gl: np.ndarray = field(init=False)
    w_gl: np.ndarray = field(init=False)
    om_gl: np.ndarray = field(init=False)

    def __post_init__(self):
        self.eta, self.D = cgl_nodes_and_diff(self.N)
        self.om = 1.0 - self.eta
        self.wq = clenshaw_curtis_weights(self.N)

        # De-aliased volume.  R = (1-eta) S with S of degree N, so R^2 has
        # degree 2N, whereas Clenshaw-Curtis on the N+1 CGL nodes is exact
        # only up to degree N.  A Gauss-Legendre rule with N+1 points is
        # exact up to degree 2N+1 and therefore integrates the interpolant
        # exactly.  P_gl carries the CGL values to the GL nodes.
        x, w = np.polynomial.legendre.leggauss(self.N + 1)
        eta_gl = 0.5 * (x + 1.0)
        self.w_gl = 0.5 * w
        self.om_gl = 1.0 - eta_gl
        E = np.eye(self.N + 1)
        self.P_gl = np.column_stack(
            [barycentric_interpolate(self.eta, E[:, j], eta_gl)
             for j in range(self.N + 1)]
        )

    # -- packing ------------------------------------------------------

    @property
    def size(self) -> int:
        return 4 * self.N + 1

    def pack(self, m, S, u, v, L) -> np.ndarray:
        return np.concatenate([m[1:], S[1:], u[1:], v[1:], [L]])

    def unpack(self, y: np.ndarray, t: float):
        N = self.N
        u0 = self.p.nozzle_velocity(t)

        m = np.empty(N + 1)
        S = np.empty(N + 1)
        u = np.empty(N + 1)
        v = np.empty(N + 1)

        m[0], S[0], u[0], v[0] = 1.0, 1.0, u0, u0 * self.p.tan_theta0
        m[1:] = y[0:N]
        S[1:] = y[N : 2 * N]
        u[1:] = y[2 * N : 3 * N]
        v[1:] = y[3 * N : 4 * N]
        return m, S, u, v, float(y[-1])

    # -- geometry and pressure ---------------------------------------

    def geometry(self, S: np.ndarray, L: float):
        DS = self.D @ S
        R = self.om * S
        R_eta = -S + self.om * DS
        R_etaeta = -2.0 * DS + self.om * (self.D @ DS)

        s = R_eta / L
        q = 1.0 + s * s
        rt = np.sqrt(q)

        J_eta = R_eta / rt - R * R_eta * R_etaeta / (L * L * q * rt)
        Jeta_over_Reta = 1.0 / rt - R * R_etaeta / (L * L * q * rt)
        return R, R_eta, J_eta, Jeta_over_Reta

    def volume(self, S: np.ndarray, L: float) -> float:
        if self.dealias:
            R = self.om_gl * (self.P_gl @ S)
            return float(L * np.dot(self.w_gl, R * R))
        R = self.om * S
        return float(L * np.dot(self.wq, R * R))

    def pressure_coefficient(self, S: np.ndarray, L: float) -> float:
        if self.volume_reference <= 0.0:
            raise ValueError("volume_reference must be set.")
        vol = self.volume(S, L)
        if vol <= 0.0:
            if self.guard == "raise":
                raise FloatingPointError(f"Nonpositive enclosed volume {vol}.")
            self.n_clamped += 1
            vol = self.floor
        return self.p.Cpmax * (
            self.p.pressure_ratio0 * self.volume_reference / vol - 1.0
        )

    # -- right-hand side ---------------------------------------------

    def rhs(self, t: float, y: np.ndarray, Cpn_fixed=None) -> np.ndarray:
        """Right-hand side of the semidiscrete system.

        On a non-physical state the default behaviour, `guard='clamp'`, is
        to floor the offending quantity at `floor` and to count the event in
        `self.n_clamped`, so that the value returned is finite and the
        integrator rejects the step in the ordinary way.  Raising instead --
        `guard='raise'`, the historical behaviour -- aborts the whole
        integration from a tentative stage, because `solve_ivp` does not
        catch exceptions from the right-hand side.  That is what prevented
        the degeneracy event from being located with a threshold below about
        1e-6 or with a loose tolerance.

        Clamping is a numerical device, not a model: a solution is
        trustworthy only up to the first *accepted* step at which clamping
        occurred.  Callers should therefore check `n_clamped` against the
        value it had before the integration, and treat any increase as a
        warning that the run has entered the region where the formulation
        has failed.  The event itself fires before that, at a positive
        threshold, which is the whole point.
        """
        m, S, u, v, L = self.unpack(y, t)

        bad = (L <= 0.0 or np.min(m) <= 0.0 or np.min(u) <= 0.0
               or S[-1] <= 0.0)
        if bad:
            if self.guard == "raise":
                raise FloatingPointError(
                    f"Nonphysical state: L={L}, min m={m.min()}, "
                    f"min u={u.min()}, S(1)={S[-1]}"
                )
            self.n_clamped += 1
            eps = self.floor
            L = max(L, eps)
            m = np.maximum(m, eps)
            u = np.maximum(u, eps)
            S = S.copy()
            S[-1] = max(S[-1], eps)

        R, R_eta, J_eta, Jeta_over_Reta = self.geometry(S, L)

        Cpn = (
            float(Cpn_fixed)
            if Cpn_fixed is not None
            else self.pressure_coefficient(S, L)
        )

        # Analytic tip kinematics: Ldot = u(1) + L v(1) / S(1).
        Ldot = u[-1] + L * v[-1] / S[-1]

        a = (u - self.eta * Ldot) / L

        Dm = self.D @ m
        Du = self.D @ u
        Dv = self.D @ v
        DS = self.D @ S

        dm = -a * Dm - (m / L) * Du
        du = (
            -a * Du
            + self.p.inv_Fr(t)
            + (J_eta - Cpn * R * R_eta) / (m * self.p.We * L)
        )
        dv = -a * Dv + (Cpn * R - Jeta_over_Reta) / (m * self.p.We)

        # S equation.  G = v + a S vanishes identically at eta = 1 by the
        # definition of Ldot, so G/(1-eta) is smooth; the tip value is the
        # L'Hopital limit -G_eta(1).
        G = v + a * S
        H = np.empty_like(G)
        H[:-1] = G[:-1] / self.om[:-1]
        H[-1] = -float((self.D @ G)[-1])
        dS = H - a * DS

        return np.concatenate([dm[1:], dS[1:], du[1:], dv[1:], [Ldot]])

    # -- initialisation from the steady solve ------------------------

    def state_from_steady(self, w: np.ndarray, steady: SteadySpectral):
        if steady.N != self.N:
            S = barycentric_interpolate(
                steady.eta, w[: steady.N + 1], self.eta
            )
            u = barycentric_interpolate(
                steady.eta,
                w[steady.N + 1 : 2 * steady.N + 2],
                self.eta,
            )
            L = float(w[-1])
        else:
            S, u, L = steady.unpack(w)

        R_eta = -S + self.om * (self.D @ S)
        v = u * R_eta / L
        m = 1.0 / u
        return self.pack(m, S, u, v, L)

    def discrete_equilibrium(self, y0: np.ndarray, tol: float = 1.0e-13):
        """Newton solve of rhs = 0 with the imposed steady Cpn."""
        Cpn = self.p.Cpmax * (self.p.pressure_ratio0 - 1.0)
        sol = root(
            lambda y: self.rhs(0.0, y, Cpn_fixed=Cpn),
            y0,
            method="hybr",
            tol=tol,
            options={"maxfev": 40000},
        )
        return sol

    # -- linear stability of the semidiscrete operator ---------------

    def jacobian(self, y: np.ndarray, Cpn_fixed=None, eps=1.0e-7):
        f0 = self.rhs(0.0, y, Cpn_fixed=Cpn_fixed)
        J = np.empty((f0.size, y.size))
        for k in range(y.size):
            h = eps * max(1.0, abs(y[k]))
            yp = y.copy()
            yp[k] += h
            J[:, k] = (self.rhs(0.0, yp, Cpn_fixed=Cpn_fixed) - f0) / h
        return J

    def jac_sparsity(self) -> np.ndarray:
        """Dense: spectral differentiation couples every node.

        Provided explicitly so callers do not mistakenly assume a banded
        structure inherited from finite differences.
        """
        return np.ones((self.size, self.size))

    # -- time integration --------------------------------------------

    def integrate(
        self,
        y0: np.ndarray,
        t_final: float,
        method: str = "Radau",
        rtol: float = 1.0e-9,
        atol: float = 1.0e-11,
        t_eval=None,
        max_step=np.inf,
        events=None,
    ):
        """Integrate in time; `max_step` matters.

        `rhs` raises FloatingPointError on a non-physical state, and
        `solve_ivp` does not catch exceptions from the right-hand side, so a
        failed tentative stage aborts the integration instead of causing the
        step to be rejected.  Bounding the step size avoids this.
        """
        return solve_ivp(
            lambda t, y: self.rhs(t, y),
            (0.0, t_final),
            y0,
            method=method,
            rtol=rtol,
            atol=atol,
            t_eval=t_eval,
            max_step=max_step,
            events=events,
            dense_output=True,
        )

    def observables(self, t: float, y: np.ndarray) -> dict:
        m, S, u, v, L = self.unpack(y, t)
        return {
            "t": float(t),
            "L": float(L),
            "Cpn": float(self.pressure_coefficient(S, L)),
            "u0": float(u[0]),
            "S_tip": float(S[-1]),
            "min_m": float(m.min()),
            "min_u": float(u.min()),
            "volume": self.volume(S, L),
        }
