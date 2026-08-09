"""Cross-validation with a structurally unrelated discretisation.

A spectral method is only believable if something built on different
principles reproduces its numbers.  This module implements a method of
lines with third-order upwind finite differences on a uniform mesh.  It
differs from the spectral solver in the three respects that matter.

  * Variables.  The spectral solver works with S = R/(1-eta), so that
    R = (1-eta) S annihilates the tip constraint *identically*.  Here R is
    integrated directly, and R(1) = 0 is preserved only because dL/dt is
    chosen so that dR/dt vanishes at the tip.  The drift of R(1) is
    therefore observable, and it is the same index-two pathology that the
    domain-adaptive schemes of Ramos carry.

  * Discretisation.  Third-order upwind-biased differences for advection,
    second-order centred differences for the derivatives in the source
    term, uniform mesh.  No global basis.

  * Absence of a removable singularity.  In the R variables the quotient
    J_eta / R_eta = q^(-1/2) - R R_etaeta / (L^2 q^(3/2)) is regular at
    eta = 1 because R vanishes there, so the 0/0 of the S formulation does
    not arise.

Governing equations in the mapped frame, with a = (u - eta dL/dt)/L:

    m_t = -a m_eta - (m/L) u_eta
    R_t = v - a R_eta
    u_t = -a u_eta + 1/Fr(t) + (J_eta - Cpn R R_eta)/(m We L)
    v_t = -a v_eta + (Cpn R - J_eta/R_eta)/(m We)
    dL/dt = u(1) - L v(1)/R_eta(1)
"""

from __future__ import annotations

import math

import numpy as np
from scipy.integrate import solve_ivp

# numpy renamed `trapz` to `trapezoid` in version 2.0 and removed the old
# name later; numpy 1.x has only `trapz`.  Binding the available one here
# keeps a single source working on both, which matters because the cluster
# used for the large runs carries numpy 1.24 while the development machine
# carries 2.4.  Hand-patching the file on one of them is what allowed the
# two copies to drift apart in the first place.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz



class UpwindFD:
    def __init__(self, M: int, p, impose_tip: bool = True):
        self.M = M
        self.p = p
        self.impose_tip = impose_tip
        self.eta = np.linspace(0.0, 1.0, M + 1)
        self.h = 1.0 / M
        self.volume_reference = 0.0

    # ---- stencils -------------------------------------------------
    def d_upwind(self, f):
        """f_eta with a > 0: third-order upwind-biased."""
        h, M = self.h, self.M
        d = np.empty_like(f)
        d[0] = (-3.0 * f[0] + 4.0 * f[1] - f[2]) / (2.0 * h)
        d[1] = (f[2] - f[0]) / (2.0 * h)
        d[2:M] = (2.0 * f[3:M + 1] + 3.0 * f[2:M]
                  - 6.0 * f[1:M - 1] + f[0:M - 2]) / (6.0 * h)
        d[M] = (11.0 * f[M] - 18.0 * f[M - 1]
                + 9.0 * f[M - 2] - 2.0 * f[M - 3]) / (6.0 * h)
        return d

    def d_central(self, f):
        h, M = self.h, self.M
        d = np.empty_like(f)
        d[0] = (-3.0 * f[0] + 4.0 * f[1] - f[2]) / (2.0 * h)
        d[1:M] = (f[2:M + 1] - f[0:M - 1]) / (2.0 * h)
        d[M] = (3.0 * f[M] - 4.0 * f[M - 1] + f[M - 2]) / (2.0 * h)
        return d

    def d2_central(self, f):
        h, M = self.h, self.M
        d = np.empty_like(f)
        d[1:M] = (f[2:M + 1] - 2.0 * f[1:M] + f[0:M - 1]) / (h * h)
        d[0] = (2.0 * f[0] - 5.0 * f[1] + 4.0 * f[2] - f[3]) / (h * h)
        d[M] = (2.0 * f[M] - 5.0 * f[M - 1] + 4.0 * f[M - 2]
                - f[M - 3]) / (h * h)
        return d

    # ---- state ----------------------------------------------------
    def pack(self, m, R, u, v, L):
        return np.concatenate([m, R, u, v, [L]])

    def unpack(self, y):
        M = self.M
        n = M + 1
        return (y[0:n], y[n:2 * n], y[2 * n:3 * n], y[3 * n:4 * n],
                float(y[-1]))

    def volume(self, R, L):
        return float(L * _trapezoid(R * R, self.eta))

    def cpn(self, R, L):
        V = self.volume(R, L)
        return self.p.Cpmax * (self.p.pressure_ratio0
                               * self.volume_reference / V - 1.0)

    # ---- right-hand side ------------------------------------------
    def rhs(self, t, y):
        p = self.p
        m, R, u, v, L = self.unpack(y)
        if L <= 0.0 or m.min() <= 0.0:
            raise FloatingPointError(f"nonphysical: L={L}, min m={m.min()}")

        Re_ = self.d_central(R)
        Ree = self.d2_central(R)
        if Re_[-1] >= 0.0:
            raise FloatingPointError(f"tip not transversal: R_eta(1)={Re_[-1]}")

        Ldot = u[-1] - L * v[-1] / Re_[-1]
        a = (u - self.eta * Ldot) / L

        s = Re_ / L
        q = 1.0 + s * s
        rt = np.sqrt(q)
        J_eta = Re_ / rt - R * Re_ * Ree / (L * L * q * rt)
        JoR = 1.0 / rt - R * Ree / (L * L * q * rt)

        Cpn = self.cpn(R, L)
        me, ue, ve = self.d_upwind(m), self.d_upwind(u), self.d_upwind(v)
        de = self.d_upwind(R)

        dm = -a * me - (m / L) * self.d_central(u)
        dR = v - a * de
        du = -a * ue + p.inv_Fr(t) + (J_eta - Cpn * R * Re_) / (m * p.We * L)
        dv = -a * ve + (Cpn * R - JoR) / (m * p.We)

        # nozzle data are imposed, not evolved
        u0 = p.nozzle_velocity(t)
        eps = 1.0e-6
        du0 = (p.nozzle_velocity(t + eps) - p.nozzle_velocity(t - eps)) / (2 * eps)
        dm[0] = 0.0
        dR[0] = 0.0
        du[0] = du0
        dv[0] = du0 * p.tan_theta0

        # Tip closure.  With `impose_tip = False` the constraint R(1) = 0 is
        # only *derived*: Ldot is chosen so that R_t(1) = 0, but the two
        # derivatives involved are discretised with different stencils
        # (d_central for Ldot, d_upwind for the advection), so dR[M] is not
        # exactly zero and R(1) drifts.  The drift feeds back into R_eta(1),
        # hence into Ldot, and the loop is unstable: this is the index-2
        # drift-off, reproduced here on purpose.  With `impose_tip = True`
        # the constraint is imposed at the last node instead.
        if self.impose_tip:
            dR[-1] = 0.0
        return self.pack(dm, dR, du, dv, Ldot)

    # ---- initial data from the spectral steady solution -----------
    def from_spectral(self, w, prob):
        """Initial data from the spectral steady solution.

        Here v = u R_eta / L is evaluated with the *spectral* derivative of
        R and only then interpolated.  Building it from the centred
        difference on the scheme's own mesh introduces an O(h^2) error in v
        which the scheme then differentiates, degrading the residual to
        O(h) and making a second-order method look first-order.
        """
        from annular_spectral import barycentric_interpolate
        Ns = prob.N
        Ss = w[: Ns + 1]
        us = w[Ns + 1: 2 * Ns + 2]
        L = float(w[-1])
        om_s = 1.0 - prob.eta
        Rs = om_s * Ss
        Re_s = -Ss + om_s * (prob.D @ Ss)          # dR/deta, spectral
        vs = us * Re_s / L

        R = barycentric_interpolate(prob.eta, Rs, self.eta)
        u = barycentric_interpolate(prob.eta, us, self.eta)
        v = barycentric_interpolate(prob.eta, vs, self.eta)
        R[-1] = 0.0
        m = 1.0 / u
        v[0] = u[0] * self.p.tan_theta0
        y = self.pack(m, R, u, v, L)
        self.volume_reference = self.volume(R, L)
        return y

    def integrate(self, y0, tf, **kw):
        opts = dict(method="DOP853", rtol=1.0e-10, atol=1.0e-12,
                    max_step=0.02)
        opts.update(kw)
        return solve_ivp(self.rhs, (0.0, tf), y0, **opts)
