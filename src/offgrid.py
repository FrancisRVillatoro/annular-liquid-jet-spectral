"""Off-grid residual of the unsteady mapped equations.

A collocation solution satisfies the equations *identically* at the N+1
nodes, so an on-grid residual certifies nothing.  What certifies a genuine
spectral approximation is the residual of the interpolating polynomial at
points where nothing was imposed.

The derivative of the degree-N interpolant is the degree-(N-1) polynomial
whose nodal values are D @ f, so interpolating D @ f to the fine grid gives
the exact off-grid derivative of the interpolant; likewise for D @ D @ f.
The time derivatives are those of the polynomial coefficients, so they
interpolate in the same way.  The residual is therefore

    r = P (dy/dt) - RHS[ interpolated fields ]

evaluated on a fine grid, and it is what should be reported alongside any
claim about the unsteady solution, in particular near a finite-time
degeneracy where the solution ceases to be analytic.
"""

from __future__ import annotations

import numpy as np

from annular_spectral import UnsteadySpectral, barycentric_interpolate


def fine_operators(uns: UnsteadySpectral, M: int = 2001,
                   eta_min: float = 0.05, eta_max: float = 1.0 - 1e-6):
    """Interpolation matrix from the CGL nodes to M interior points.

    The window excludes a neighbourhood of the nozzle.  At eta = 0 the four
    boundary values are imposed algebraically and the equations are *not*
    collocated there, so the polynomial is under no obligation to satisfy
    them at that node; the residual peaks there and decays inward over the
    first CGL spacing.  Including it would report a boundary artefact
    rather than the quality of the interior approximation.
    """
    eta_f = np.linspace(eta_min, eta_max, M)
    E = np.eye(uns.N + 1)
    P = np.column_stack(
        [barycentric_interpolate(uns.eta, E[:, j], eta_f)
         for j in range(uns.N + 1)]
    )
    return eta_f, P


def offgrid_residual(uns: UnsteadySpectral, t: float, y: np.ndarray,
                     eta_f=None, P=None, M: int = 2001):
    """Max-norm of the off-grid residual, one entry per equation.

    Returns dict with keys 'm', 'S', 'u', 'v' and 'max'.
    """
    if P is None:
        eta_f, P = fine_operators(uns, M)

    N = uns.N
    m, S, u, v, L = uns.unpack(y, t)
    dy = uns.rhs(t, y)

    # nodal time derivatives, including the algebraic nozzle values
    dm = np.concatenate([[0.0], dy[0:N]])
    dS = np.concatenate([[0.0], dy[N:2 * N]])
    du_0 = (uns.p.nozzle_velocity(t + 1e-6)
            - uns.p.nozzle_velocity(t - 1e-6)) / 2e-6
    du = np.concatenate([[du_0], dy[2 * N:3 * N]])
    dv = np.concatenate([[du_0 * uns.p.tan_theta0], dy[3 * N:4 * N]])
    Ldot = float(dy[-1])

    D = uns.D
    # fields and their derivatives, off grid
    mf, Sf, uf, vf = P @ m, P @ S, P @ u, P @ v
    Dm, DS, Du, Dv = P @ (D @ m), P @ (D @ S), P @ (D @ u), P @ (D @ v)
    DDS = P @ (D @ (D @ S))

    om = 1.0 - eta_f
    R = om * Sf
    R_eta = -Sf + om * DS
    R_etaeta = -2.0 * DS + om * DDS

    s = R_eta / L
    q = 1.0 + s * s
    rt = np.sqrt(q)
    J_eta = R_eta / rt - R * R_eta * R_etaeta / (L * L * q * rt)
    Jeta_over_Reta = 1.0 / rt - R * R_etaeta / (L * L * q * rt)

    Cpn = uns.pressure_coefficient(S, L)
    a = (uf - eta_f * Ldot) / L

    rhs_m = -a * Dm - (mf / L) * Du
    rhs_u = (-a * Du + uns.p.inv_Fr(t)
             + (J_eta - Cpn * R * R_eta) / (mf * uns.p.We * L))
    rhs_v = -a * Dv + (Cpn * R - Jeta_over_Reta) / (mf * uns.p.We)
    rhs_S = (vf + a * Sf) / om - a * DS

    out = {
        "m": float(np.max(np.abs(P @ dm - rhs_m))),
        "S": float(np.max(np.abs(P @ dS - rhs_S))),
        "u": float(np.max(np.abs(P @ du - rhs_u))),
        "v": float(np.max(np.abs(P @ dv - rhs_v))),
    }
    out["max"] = max(out[k] for k in ("m", "S", "u", "v"))
    return out


def modal_tail(uns: UnsteadySpectral, y: np.ndarray, t: float = 0.0):
    """Ratio of the last two Chebyshev coefficients to the largest, per field.

    A resolved analytic solution has a tail at round-off; a tail that
    stagnates is the signature of a loss of regularity.
    """
    m, S, u, v, L = uns.unpack(y, t)
    out = {}
    for name, f in (("m", m), ("S", S), ("u", u), ("v", v)):
        # values at CGL nodes -> Chebyshev coefficients via the DCT-I
        g = f[::-1]                       # to x = cos(k pi / N) ordering
        n = uns.N
        ext = np.concatenate([g, g[-2:0:-1]])
        c = np.real(np.fft.fft(ext))[: n + 1] / n
        c[0] *= 0.5
        c[n] *= 0.5
        out[name] = float(np.max(np.abs(c[-2:])) / np.max(np.abs(c)))
    return out
