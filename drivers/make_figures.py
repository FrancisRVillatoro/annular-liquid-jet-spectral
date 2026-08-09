"""Figures for the manuscript.  All labels in English.

No text inside the axes duplicates the caption: panels carry only the
axis labels and, where a key is strictly necessary to identify curves,
a legend placed clear of the data.  Everything else belongs in the
caption.

Usage:
    python3 make_figures.py            # all
    python3 make_figures.py 1 3        # selected
"""

from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))



import sys

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from annular_spectral import Parameters, UnsteadySpectral, solve_steady
from annular_thickness import ThickParameters, reference_solve
from offgrid import fine_operators, offgrid_residual

mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.1,
    "lines.markersize": 4,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "legend.frameon": False,
    "figure.dpi": 140,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

# Only external data used anywhere: Table IV of Ramos, Comput. Mech. 11
# (1993), for Fr = 10, We = 50, theta0 = 0, Cpn = 0.5.
TABLE_1993 = {0.005: 17.604, 0.050: 18.239, 0.100: 18.609}

BODY = dict(Fr=10.0, We=50.0, theta0_deg=0.0, Cpmax=1.0,
            pressure_ratio0=1.0, amplitude=0.0, ramp_cycles=0.0,
            body_amplitude=0.5, St_g=0.5)

ONE_COL, TWO_COL = 3.37, 6.69      # AIP figure widths, inches


def panel_label(ax, text, x=0.03, y=0.95):
    ax.text(x, y, text, transform=ax.transAxes, ha="left", va="top",
            fontsize=9)


def save(fig, stem):
    for ext in ("pdf", "png"):
        fig.savefig(f"{stem}.{ext}")
    print(f"  wrote {stem}.pdf / .png")


# ======================================================================
def figure1():
    print("Figure 1: convergence")

    p = Parameters(Fr=10, We=50, theta0_deg=0)
    Lstar = float(solve_steady(96, p)[0][-1])
    Na = np.array([6, 8, 10, 12, 14, 16, 20, 24])
    ea = np.array([abs(float(solve_steady(int(n), p)[0][-1]) - Lstar)
                   for n in Na])
    print("   steady done")

    pf = Parameters(Fr=10, We=50, theta0_deg=0, amplitude=0.1, St=0.1,
                    ramp_cycles=2.0)
    T, NCYC = pf.period, 9
    Nb, eb = [16, 24, 32, 48], []
    for N in Nb:
        w, prob = solve_steady(N, Parameters(Fr=10, We=50, theta0_deg=0))
        op = UnsteadySpectral(N, pf, dealias=True)
        y0 = op.state_from_steady(w, prob)
        op.volume_reference = op.volume(op.unpack(y0, 0.0)[1], float(y0[-1]))
        sol = op.integrate(y0, NCYC * T, method="DOP853", rtol=1e-10,
                           atol=1e-12, max_step=0.05)
        ef, P = fine_operators(op)
        eb.append(offgrid_residual(op, sol.t[-1], sol.y[:, -1], ef, P)["max"])
        print(f"   forced N={N} done")
    eb = np.array(eb)

    fig, ax = plt.subplots(1, 2, figsize=(TWO_COL, 2.5))

    ax[0].semilogy(Na, ea, "o-", color="0.15")
    ax[0].set_xlabel(r"$N$")
    ax[0].set_ylabel(r"$|L_N-L^{*}|$")
    ax[0].set_xlim(4, 26)
    panel_label(ax[0], "(a)", x=0.90, y=0.95)

    ax[1].semilogy(Nb, eb, "s-", color="0.15")
    ax[1].set_xlabel(r"$N$")
    ax[1].set_ylabel(r"$\|r\|_{\infty}$")
    ax[1].set_xlim(12, 52)
    panel_label(ax[1], "(b)", x=0.90, y=0.95)

    for a in ax:
        a.grid(True, which="major", lw=0.3, color="0.85")
    fig.tight_layout()
    save(fig, "fig1_convergence")
    plt.close(fig)


# ======================================================================
def figure2():
    """Loss of transversality: coarsest and finest apart from the family."""
    print("Figure 2: loss of transversality")
    p = Parameters(**BODY)
    groups = [[16, 64], [24, 32, 48]]
    solid = dict(color="0.0")
    dashed = dict(color="0.0", dashes=(5, 2))
    dotted = dict(color="0.0", dashes=(1.3, 1.3))
    styles = [solid, dashed, dotted]

    curves = {}
    for N in (16, 24, 32, 48, 64):
        w, prob = solve_steady(N, Parameters(Fr=10, We=50, theta0_deg=0))
        op = UnsteadySpectral(N, p, dealias=True)
        y0 = op.state_from_steady(w, prob)
        op.volume_reference = op.volume(op.unpack(y0, 0.0)[1], float(y0[-1]))

        def ev(t, y):
            return float(op.unpack(y, t)[1][-1]) - 1.0e-6
        ev.terminal = True
        ev.direction = -1
        te = np.linspace(0.0, 14.0, 2801)
        sol = op.integrate(y0, 14.0, method="DOP853", rtol=1e-10, atol=1e-12,
                           t_eval=te, max_step=0.02, events=ev)
        S1 = np.array([op.unpack(sol.y[:, k], sol.t[k])[1][-1]
                       for k in range(sol.t.size)])
        tev = float(sol.t_events[0][0]) if sol.t_events[0].size else None
        curves[N] = (sol.t, S1, tev)
        print(f"   N={N}: t_end={sol.t[-1]:.4f}, min S(1)={S1.min():.4f}")

    fig, ax = plt.subplots(1, 2, figsize=(TWO_COL, 2.6))
    for col, group in enumerate(groups):
        for j, N in enumerate(group):
            t, S1, tev = curves[N]
            # the coarsest resolution is the only one plotted dotted in (a)
            st = styles[j] if col == 1 else (dotted if N == 16 else solid)
            ax[col].plot(t, S1, lw=1.0, label=fr"$N={N}$", **st)
            if tev is not None:
                ax[col].plot(tev, 0.0, "o", color="0.0", ms=3.5, zorder=5)
        ax[col].axhline(0.0, color="0.7", lw=0.5)
        ax[col].set_xlabel(r"$t$")
        ax[col].set_ylabel(r"$S(\tau,1)$")
        ax[col].set_xlim(0, 13)
        ax[col].set_ylim(-0.15, 4.6)
        ax[col].legend(loc="upper left", handlelength=2.2)
        ax[col].grid(True, lw=0.3, color="0.9")
        panel_label(ax[col], "(a)" if col == 0 else "(b)", x=0.92, y=0.95)

    fig.tight_layout()
    save(fig, "fig2_transversality")
    plt.close(fig)



def _sqrt_law_coefficient(L0, path=None):
    """Coefficient c of L(0) - L(beta) = c beta^(1/2), from the small-beta
    table if it has been computed, otherwise from a short local sweep."""
    import csv
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "data", "table10_beta_law.csv")
    if os.path.exists(path):
        with open(path, newline="") as fh:
            rows = list(csv.DictReader(fh))
        b = np.array([float(r["beta"]) for r in rows])
        d = np.array([float(r["L0_minus_L_We50"]) for r in rows])
        o = np.argsort(b)
        b, d = b[o][:20], d[o][:20]
        A = np.vstack([np.ones_like(b), np.sqrt(b)]).T
        return float(np.linalg.lstsq(A, d / np.sqrt(b), rcond=None)[0][0])
    from annular_thickness import ThickParameters, reference_solve
    def L(beta):
        p = ThickParameters(Fr=10.0, We=50.0, theta0_deg=0.0, beta=beta,
                            Cpmax=1.0, pressure_ratio0=1.5)
        return reference_solve(p, z_max=400.0)[0]
    bb = np.logspace(-6, -4, 5)
    return float(np.mean([(L0 - L(x)) / x ** 0.5 for x in bb]))


# ======================================================================
def figure3():
    print("Figure 3: finite thickness")

    def L_of(beta, We, Cpn):
        tp = ThickParameters(Fr=10.0, We=We, theta0_deg=0.0, beta=beta,
                             Cpmax=1.0, pressure_ratio0=1.0 + Cpn)
        return reference_solve(tp, z_max=400.0)[0]

    betas = np.concatenate([[0.0], np.linspace(0.002, 0.20, 40)])
    L50 = np.array([L_of(b, 50.0, 0.5) for b in betas])
    L25 = np.array([L_of(b, 25.0, 0.0) for b in betas])
    print("   curves done")

    fig, ax = plt.subplots(1, 2, figsize=(TWO_COL, 2.5))

    bR = np.array(sorted(TABLE_1993))
    LR = np.array([TABLE_1993[b] for b in bR])
    ax[0].plot(betas, L50, "-", color="0.15")
    ax[0].plot(bR, LR, "s", mfc="none", color="0.15")
    ax[0].set_xlabel(r"$\beta$")
    ax[0].set_ylabel(r"$L$")
    panel_label(ax[0], "(a)", x=0.90, y=0.95)

    # The square-root law is fitted on the small-beta data of
    # data/table10_beta_law.csv when that file is present, which is where
    # the asymptotic statement lives; anchoring the fit at the largest
    # beta plotted would test it where it is least valid.
    c = _sqrt_law_coefficient(L50[0]) / L50[0]
    ax[1].plot(betas, L50 / L50[0], "-", color="0.15")
    ax[1].plot(betas, L25 / L25[0], "--", color="0.45")
    ax[1].plot(betas, 1.0 - c * np.sqrt(betas), ":", color="0.0", lw=0.9)
    ax[1].set_xlabel(r"$\beta$")
    ax[1].set_ylabel(r"$L(\beta)/L(0)$")
    panel_label(ax[1], "(b)", x=0.90, y=0.95)

    for a in ax:
        a.grid(True, lw=0.3, color="0.9")
    fig.tight_layout()
    save(fig, "fig3_thickness")
    plt.close(fig)



# ======================================================================
def figure4():
    """Steady interface profiles and the square-root retreat of the tip."""
    print("Figure 4: steady profiles")
    from annular_thickness import ThickParameters, reference_solve

    betas = (0.0, 0.05, 0.20)
    dash = {0.0: dict(), 0.05: dict(dashes=(5, 2)), 0.20: dict(dashes=(1.5, 1.5))}
    gray = {0.0: "0.0", 0.05: "0.35", 0.20: "0.6"}

    fig, ax = plt.subplots(1, 2, figsize=(TWO_COL, 2.5))
    for beta in betas:
        p = ThickParameters(Fr=10.0, We=50.0, theta0_deg=0.0, beta=beta,
                            Cpmax=1.0, pressure_ratio0=1.5)
        L, sol = reference_solve(p, z_max=400.0)
        z = np.linspace(0.0, L, 4001)
        Y = sol.sol(z)
        R, u = Y[0], Y[1]
        m = 1.0 / u
        Ri = R - 0.5 * beta * m / R if beta > 0 else R
        Re = R + 0.5 * beta * m / R if beta > 0 else R
        for a in ax:
            a.plot(z, Ri, color=gray[beta], lw=1.0, **dash[beta])
            if beta > 0:
                a.plot(z, Re, color=gray[beta], lw=1.0, **dash[beta])
        print(f"   beta={beta}: L={L:.5f}")

    ax[0].set_xlabel(r"$z$")
    ax[0].set_ylabel(r"$R_i,\;R_e$")
    ax[0].set_xlim(0, 18)
    ax[0].set_ylim(0, 1.35)
    panel_label(ax[0], "(a)", x=0.90, y=0.95)

    ax[1].set_xlabel(r"$z$")
    ax[1].set_ylabel(r"$R_i,\;R_e$")
    ax[1].set_xlim(14.5, 17.6)
    ax[1].set_ylim(0, 0.35)
    panel_label(ax[1], "(b)", x=0.90, y=0.95)

    for a in ax:
        a.grid(True, lw=0.3, color="0.9")
    fig.tight_layout()
    save(fig, "fig4_profiles")
    plt.close(fig)


# ======================================================================
def figure5():
    """Interface approaching the loss of transversality, in two groups."""
    print("Figure 5: interface near the degeneracy")
    from annular_spectral import barycentric_interpolate
    N = 64
    p = Parameters(**BODY)
    w, prob = solve_steady(N, Parameters(Fr=10, We=50, theta0_deg=0))
    op = UnsteadySpectral(N, p, dealias=True)
    y0 = op.state_from_steady(w, prob)
    op.volume_reference = op.volume(op.unpack(y0, 0.0)[1], float(y0[-1]))

    groups = [[0.0, 8.0, 10.0], [10.5, 10.70, 10.7665]]
    times = groups[0] + groups[1]
    sol = op.integrate(y0, times[-1], method="DOP853", rtol=1e-11,
                       atol=1e-13, t_eval=times, max_step=0.02)
    ef = np.linspace(0.0, 1.0, 1601)

    styles = [dict(color="0.0"),
              dict(color="0.0", dashes=(5, 2)),
              dict(color="0.0", dashes=(1.3, 1.3))]

    profiles = []
    for k, t in enumerate(sol.t):
        m, S, u, v, L = op.unpack(sol.y[:, k], t)
        Sf = barycentric_interpolate(op.eta, S, ef)
        profiles.append((t, ef * L, (1.0 - ef) * Sf))
        print(f"   t={t:.4f}  L={L:.5f}  S(1)={S[-1]:.4f}")

    fig, ax = plt.subplots(2, 2, figsize=(TWO_COL, 4.4))
    labels = [["(a)", "(b)"], ["(c)", "(d)"]]
    for col in (0, 1):
        sel = profiles[3 * col:3 * col + 3]
        for j, (t, z, R) in enumerate(sel):
            lab = f"$t={t:g}$" if t != 10.7665 else r"$t=t^{*}$"
            ax[0, col].plot(z, R, lw=1.0, label=lab, **styles[j])
            ax[1, col].plot(z, R, lw=1.0, **styles[j])
        ax[0, col].set_xlim(0, 14)
        ax[0, col].set_ylim(0, 1.05)
        ax[1, col].set_xlim(11.6, 13.5)
        ax[1, col].set_ylim(0, 0.16)
        ax[0, col].legend(loc="upper right", handlelength=2.2)
        for row in (0, 1):
            ax[row, col].set_xlabel(r"$z$")
            ax[row, col].set_ylabel(r"$R$")
            ax[row, col].grid(True, lw=0.3, color="0.9")
            # in the top panels the interface starts at R = 1, so the
            # label is dropped to R = 0.9 to keep it clear of the curve
            ypos = 0.9 / 1.05 if row == 0 else 0.95
            panel_label(ax[row, col], labels[row][col], x=0.055, y=ypos)

    fig.tight_layout()
    save(fig, "fig5_interface")
    plt.close(fig)


if __name__ == "__main__":
    which = sys.argv[1:] or ["1", "2", "3", "4", "5"]
    if "1" in which:
        figure1()
    if "2" in which:
        figure2()
    if "3" in which:
        figure3()
    if "4" in which:
        figure4()
    if "5" in which:
        figure5()
