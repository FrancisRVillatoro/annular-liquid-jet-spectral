"""Row-level task registry for the tables of the paper.

Each row of each table is an independent computation. This module exposes
them as a flat list of tasks so that they can be run serially by
`make_tables.py`, or one per array task on a cluster by `run_task.py` and
reassembled by `merge_tables.py`.

The registry is the single definition of what each table contains; the
serial and the parallel paths call exactly the same functions, so the two
cannot drift apart.
"""

from __future__ import annotations
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src"))

from annular_spectral import (Parameters, UnsteadySpectral,  # noqa: E402
                              solve_steady)
from annular_thickness import ThickParameters, reference_solve  # noqa: E402
from offgrid import fine_operators, offgrid_residual  # noqa: E402
from upwind_fd import UpwindFD  # noqa: E402

BODY = dict(Fr=10.0, We=50.0, theta0_deg=0.0, Cpmax=1.0,
            pressure_ratio0=1.0, amplitude=0.0, ramp_cycles=0.0,
            body_amplitude=0.5, St_g=0.5)


def thick_L(Fr, We, beta, Cpn, theta0=0.0):
    p = ThickParameters(Fr=Fr, We=We, theta0_deg=theta0, beta=beta,
                        Cpmax=1.0, pressure_ratio0=1.0 + Cpn)
    return reference_solve(p, z_max=400.0)[0]


# ------------------------------------------------------------------ I
_L_STAR = None


def _lstar():
    global _L_STAR
    if _L_STAR is None:
        _L_STAR = float(solve_steady(96, Parameters(Fr=10, We=50,
                                                    theta0_deg=0))[0][-1])
    return _L_STAR


def t1_row(N):
    L = float(solve_steady(N, Parameters(Fr=10, We=50, theta0_deg=0))[0][-1])
    return [N, f"{L:.14f}", f"{abs(L - _lstar()):.6e}"]


# ----------------------------------------------------------------- II
_REF92 = [("Fr=10, We=50, theta0=0", dict(Fr=10, We=50, theta0_deg=0), 12.5590),
          ("Fr=1e4, We=50, theta0=0", dict(Fr=10000, We=50, theta0_deg=0), 9.8652),
          ("Fr=10, We=100, theta0=0", dict(Fr=10, We=100, theta0_deg=0), 19.0452),
          ("Fr=10, We=50, theta0=-15", dict(Fr=10, We=50, theta0_deg=-15), 3.8236),
          ("Fr=10, We=50, theta0=+15", dict(Fr=10, We=50, theta0_deg=15), 37.3466),
          ("Fr=10, We=50, pi/pe=0.5",
           dict(Fr=10, We=50, theta0_deg=0, pressure_ratio0=0.5), 10.2816)]


def t2_row(k):
    name, kw, Lr = _REF92[k]
    pp = Parameters(**kw)
    w, prob = solve_steady(32, pp)
    L = float(w[-1])
    return [name, f"{pp.critical_margin():.4f}", f"{L:.8f}", f"{Lr:.4f}",
            f"{abs(L - Lr) / Lr:.3e}",
            f"{np.max(np.abs(prob.residual(w))):.2e}"]


# ---------------------------------------------------------------- III
def t3_row(We):
    Lref = thick_L(10.0, We, 0.0, 0.0)
    try:
        L = float(solve_steady(64, Parameters(Fr=10.0, We=We,
                                              theta0_deg=0.0))[0][-1])
        return [f"{We:.4f}", f"{We - 1:.4f}", f"{L:.10f}", f"{Lref:.10f}",
                f"{abs(L - Lref) / Lref:.2e}"]
    except Exception:
        return [f"{We:.4f}", f"{We - 1:.4f}", "not converged",
                f"{Lref:.10f}", ""]


# ----------------------------------------------------------------- IV
def t4_row(N, method):
    pf = Parameters(Fr=10, We=50, theta0_deg=0, amplitude=0.1, St=0.1,
                    ramp_cycles=2.0)
    T, NCYC = pf.period, 9
    w, prob = solve_steady(N, Parameters(Fr=10, We=50, theta0_deg=0))
    op = UnsteadySpectral(N, pf, dealias=True)
    y0 = op.state_from_steady(w, prob)
    op.volume_reference = op.volume(op.unpack(y0, 0.0)[1], float(y0[-1]))
    te = np.linspace(0.0, NCYC * T, 40 * NCYC + 1)
    sol = op.integrate(y0, NCYC * T, method=method, rtol=1e-10, atol=1e-12,
                       t_eval=te, max_step=0.05)
    obs = [op.observables(t, sol.y[:, k]) for k, t in enumerate(sol.t)]
    tt = np.array([o["t"] for o in obs])
    LL = np.array([o["L"] for o in obs])
    UU = np.array([o["u0"] for o in obs])
    m = tt >= (NCYC - 3) * T
    wf = 2 * np.pi * pf.St
    M = np.column_stack([np.ones(m.sum()), np.cos(wf * tt[m]),
                         np.sin(wf * tt[m])])
    c, *_ = np.linalg.lstsq(M, LL[m], rcond=None)
    cu, *_ = np.linalg.lstsq(M, UU[m], rcond=None)
    ph = np.arctan2(-c[2], c[1]) - np.arctan2(-cu[2], cu[1])
    lag = ((-ph) % (2 * np.pi)) / (2 * np.pi) * T
    ef, P = fine_operators(op)
    res = offgrid_residual(op, sol.t[-1], sol.y[:, -1], ef, P)["max"]
    return [N, method, f"{c[0]:.8f}", f"{np.hypot(c[1], c[2]) / c[0]:.5f}",
            f"{lag:.4f}", f"{res:.3e}"]


# ------------------------------------------------------------------ V
def t5_row(beta):
    return [f"{beta:.3f}", f"{thick_L(10.0, 50.0, beta, 0.5):.6f}",
            f"{thick_L(10.0, 25.0, beta, 0.0):.6f}"]


# ----------------------------------------------------------------- VI
_REF93 = [("pi/pe=0.5", 10.0, 50.0, 0.05, -0.50, 11.030),
          ("pi/pe=1.0", 10.0, 50.0, 0.05, 0.00, 13.388),
          ("Cpmax=1, r=1.05", 10.0, 50.0, 0.05, 0.05, 13.712),
          ("Cpmax=5, r=1.05", 10.0, 50.0, 0.05, 0.25, 15.278),
          ("Cpmax=10, r=1.05", 10.0, 50.0, 0.05, 0.50, 18.239),
          ("beta=0.005", 10.0, 50.0, 0.005, 0.50, 17.604),
          ("beta=0.1", 10.0, 50.0, 0.10, 0.50, 18.609),
          ("Fr=1000", 1000.0, 50.0, 0.05, 0.50, 13.930),
          ("Fr=infinity", 1e12, 50.0, 0.05, 0.50, 13.872),
          ("We=5", 10.0, 5.0, 0.05, 0.50, 4.511),
          ("We=75", 10.0, 75.0, 0.05, 0.50, 23.289)]


def t6_row(k):
    name, Fr, We, beta, Cpn, Lr = _REF93[k]
    Lm = thick_L(Fr, We, 0.0, Cpn)
    Lb = thick_L(Fr, We, beta, Cpn)
    return [name, f"{Cpn:.2f}", f"{Lm:.5f}", f"{Lb:.5f}", f"{Lr:.3f}",
            f"{Lr / Lm:.3f}"]


# ---------------------------------------------------------------- VII
def t7_row(St):
    G_L, G_C, AMP, N = 0.560359, 0.582084, 0.02, 32
    w, prob = solve_steady(N, Parameters(Fr=10, We=50, theta0_deg=0))
    p = Parameters(Fr=10, We=50, theta0_deg=0, amplitude=AMP, St=St,
                   ramp_cycles=2.0)
    T = p.period
    op = UnsteadySpectral(N, p, dealias=True)
    y0 = op.state_from_steady(w, prob)
    op.volume_reference = op.volume(op.unpack(y0, 0.0)[1], float(y0[-1]))
    NC = max(8, int(np.ceil(70.0 / T)) + 5)
    te = np.linspace(0.0, NC * T, 60 * NC + 1)
    sol = op.integrate(y0, NC * T, method="DOP853", rtol=1e-11, atol=1e-13,
                       t_eval=te, max_step=0.05)
    obs = [op.observables(t, sol.y[:, k]) for k, t in enumerate(sol.t)]
    tt = np.array([o["t"] for o in obs])
    LL = np.array([o["L"] for o in obs])
    CC = np.array([o["Cpn"] for o in obs])
    m = tt >= (NC - 3) * T
    wf = 2 * np.pi * St
    M = np.column_stack([np.ones(m.sum()), np.cos(wf * tt[m]),
                         np.sin(wf * tt[m])])
    cL, *_ = np.linalg.lstsq(M, LL[m], rcond=None)
    cC, *_ = np.linalg.lstsq(M, CC[m], rcond=None)
    gL = np.hypot(cL[1], cL[2]) / (AMP * cL[0])
    gC = np.hypot(cC[1], cC[2]) / AMP
    return [f"{St:.3f}", f"{gL:.5f}", f"{gL / G_L:.3f}", f"{gC:.5f}",
            f"{gC / G_C:.3f}"]


# --------------------------------------------------------------- VIII
def _event_time(N, eps, method="DOP853", rtol=1e-10, atol=1e-12):
    p = Parameters(**BODY)
    w, prob = solve_steady(N, Parameters(Fr=10, We=50, theta0_deg=0))
    op = UnsteadySpectral(N, p, dealias=True)
    y0 = op.state_from_steady(w, prob)
    op.volume_reference = op.volume(op.unpack(y0, 0.0)[1], float(y0[-1]))

    def ev(t, y):
        return float(op.unpack(y, t)[1][-1]) - eps
    ev.terminal, ev.direction = True, -1
    try:
        sol = op.integrate(y0, 60.0, method=method, rtol=rtol, atol=atol,
                           max_step=0.02, events=ev)
    except FloatingPointError as exc:
        return f"FAILED: {exc}"
    if sol.t_events[0].size:
        return float(sol.t_events[0][0])
    if sol.status != 0:
        return f"FAILED: integrator status {sol.status}"
    return None


def t8_row(study, eps, N, method, rtol, atol):
    t = _event_time(N, eps, method, rtol, atol)
    if t is None:
        cell = "not detected"
    elif isinstance(t, str):
        cell = t
    else:
        cell = f"{t:.9f}"
    return [study, f"{eps:.0e}", N, method, f"{rtol:.0e}", cell]


# ----------------------------------------------------------------- IX
def t9_row(M):
    """Degeneracy time from the upwind finite-difference scheme.

    Three outcomes are possible and are reported distinctly, because they
    mean different things: the event fires, giving t*; the integration
    completes without the event, which is the genuine statement that a mesh
    this coarse does not see the degeneracy; or the right-hand side raises,
    which is a failure of the run and not a property of the mesh.  Merging
    the last two under a single label would hide a failed job.
    """
    p = Parameters(**BODY)
    w, prob = solve_steady(64, Parameters(Fr=10, We=50, theta0_deg=0))
    op = UpwindFD(M, p, impose_tip=True)
    y0 = op.from_spectral(w, prob)

    def ev(t, y):
        return -op.d_central(op.unpack(y)[1])[-1] - 1.0e-2
    ev.terminal, ev.direction = True, -1
    try:
        sol = op.integrate(y0, 30.0, events=ev)
    except FloatingPointError as exc:
        return [M, f"FAILED: {type(exc).__name__}: {exc}"]
    te = sol.t_events[0]
    if te.size:
        return [M, f"{float(te[0]):.6f}"]
    if sol.status != 0:
        return [M, f"FAILED: integrator status {sol.status}: {sol.message}"]
    R = op.unpack(sol.y[:, -1])[1]
    return [M, f"no event; min S(1) reached "
               f"{-op.d_central(R)[-1]:.4f} by t={sol.t[-1]:.3f}"]



# ------------------------------------------------- X (beta, small range)
def t10_row(beta):
    """One point of the small-beta law L(0) - L(beta) = c beta^(1/2).

    The membrane values are recomputed in every row so that the task is
    self-contained; they cost a few hundredths of a second.
    """
    L0_a = thick_L(10.0, 50.0, 0.0, 0.5)
    L0_b = thick_L(10.0, 25.0, 0.0, 0.0)
    La = thick_L(10.0, 50.0, beta, 0.5)
    Lb = thick_L(10.0, 25.0, beta, 0.0)
    return [f"{beta:.8e}", f"{La:.10f}", f"{Lb:.10f}",
            f"{L0_a - La:.10e}", f"{L0_b - Lb:.10e}",
            f"{(L0_a - La) / beta ** 0.5:.8f}",
            f"{(L0_b - Lb) / beta ** 0.5:.8f}"]


# --------------------------------------------- XI (dense Strouhal sweep)
def t11_row(St):
    """One point of the dense frequency sweep; same protocol as table VII.

    The Strouhal number is written with six decimals rather than the three
    of Table VII: the grid spacing here is about 5e-3, so three decimals
    happen not to collide, but they do not identify the grid point either,
    and a deposited table should be readable without knowing how it was
    generated.
    """
    row = t7_row(St)
    row[0] = f"{St:.6f}"
    return row


_BETAS = list(np.logspace(-4.0, -2.0, 100))
# The deposited data/table11_frequency_sweep.csv was produced with this
# range; it is kept exactly so that the deposited code reproduces the
# deposited data.  The last point, St = 0.410, is the first at which the
# second degeneracy terminates the run before a limit cycle is reached, and
# it is reported as such rather than removed.
_ST_DENSE = list(np.linspace(0.005, 0.410, 80))

# ---------------------------------------------------------- registry
TABLES = {
    "table1_spectral_convergence": {
        "header": ["N", "L", "abs_error_vs_N96"],
        "tasks": [(t1_row, (N,)) for N in
                  (6, 8, 10, 12, 14, 16, 20, 24, 32, 48)],
    },
    "table2_ramos1992_membrane": {
        "header": ["case", "critical_margin", "L_present", "L_ramos1992",
                   "relative_difference", "newton_residual"],
        "tasks": [(t2_row, (k,)) for k in range(len(_REF92))],
    },
    "table3_near_critical": {
        "header": ["We", "critical_margin", "L_collocation_N64", "L_march",
                   "relative_difference"],
        "tasks": [(t3_row, (We,)) for We in
                  (1.5, 1.2, 1.1, 1.05, 1.02, 1.01, 1.005, 1.001)],
    },
    "table4_forced_response": {
        "header": ["N", "integrator", "mean_L", "amplitude_ratio", "lag_L",
                   "offgrid_residual"],
        "tasks": [(t4_row, a) for a in ((16, "DOP853"), (24, "DOP853"),
                                        (32, "DOP853"), (48, "DOP853"),
                                        (24, "Radau"))],
    },
    "table5_thickness": {
        "header": ["beta", "L_We50_Cpn0.5", "L_We25_Cpn0"],
        "tasks": [(t5_row, (b,)) for b in (0.0, 0.005, 0.01, 0.05, 0.1, 0.2)],
    },
    "table6_ramos1993_thickness": {
        "header": ["case", "Cpn", "L_membrane", "L_finite_beta",
                   "L_ramos1993", "ratio_ref_over_membrane"],
        "tasks": [(t6_row, (k,)) for k in range(len(_REF93))],
    },
    "table7_frequency_response": {
        "header": ["St", "gain_L", "gain_L_normalised", "gain_Cpn",
                   "gain_Cpn_normalised"],
        "tasks": [(t7_row, (St,)) for St in
                  (0.005, 0.010, 0.020, 0.050, 0.080, 0.100, 0.125, 0.150,
                   0.200, 0.300, 0.400)],
    },
    "table8_degeneracy_time": {
        "header": ["study", "event_threshold", "N", "integrator", "rtol",
                   "t_star"],
        "tasks": ([(t8_row, ("threshold", e, 32, "DOP853", 1e-10, 1e-12))
                   for e in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6)]
                  + [(t8_row, ("resolution", 1e-6, N, "DOP853", 1e-10, 1e-12))
                     for N in (16, 20, 24, 32, 48, 64, 96)]
                  + [(t8_row, ("integrator", 1e-6, 32, m, r, a))
                     for m, r, a in (("DOP853", 1e-10, 1e-12),
                                     ("DOP853", 1e-12, 1e-14),
                                     ("Radau", 1e-9, 1e-11))]),
    },
    "table9_cross_check": {
        "header": ["M", "t_star_upwind_fd"],
        "tasks": [(t9_row, (M,)) for M in (80, 160, 320, 640)],
    },
    "table10_beta_law": {
        "header": ["beta", "L_We50_Cpn0.5", "L_We25_Cpn0",
                   "L0_minus_L_We50", "L0_minus_L_We25",
                   "ratio_over_sqrt_beta_We50",
                   "ratio_over_sqrt_beta_We25"],
        "tasks": [(t10_row, (b,)) for b in _BETAS],
    },
    "table11_frequency_sweep": {
        "header": ["St", "gain_L", "gain_L_normalised", "gain_Cpn",
                   "gain_Cpn_normalised"],
        "tasks": [(t11_row, (St,)) for St in _ST_DENSE],
    },
}



# ======================================================================
# Supporting studies added after the first submission draft.  Their
# purpose is to close, in advance, the questions that a referee would
# reasonably ask about claims resting on a single parameter value or on a
# coarse sample.  The degeneracy events below use a threshold of 1e-4
# rather than 1e-6: with the clamping guard the integration no longer
# aborts, but the step size still collapses below about 1e-6 in the more
# strongly forced cases, and 1e-4 costs only about 1e-5 in t* (Table VIII).
# ======================================================================

EPS_EVENT = 1.0e-4


def _degeneracy(N, p, eps=EPS_EVENT, tf=60.0, seed_theta0=0.0):
    """Time at which S(tau,1) falls below eps, with diagnostics."""
    base = Parameters(Fr=p.Fr, We=p.We, theta0_deg=seed_theta0)
    w, prob = solve_steady(N, base)
    op = UnsteadySpectral(N, p, dealias=True)
    y0 = op.state_from_steady(w, prob)
    op.volume_reference = op.volume(op.unpack(y0, 0.0)[1], float(y0[-1]))

    def ev(t, y):
        return float(op.unpack(y, t)[1][-1]) - eps
    ev.terminal, ev.direction = True, -1
    sol = op.integrate(y0, tf, method="DOP853", rtol=1.0e-10, atol=1.0e-12,
                       max_step=0.02, events=ev)
    smin = min(float(op.unpack(sol.y[:, k], sol.t[k])[1][-1])
               for k in range(0, sol.t.size, max(1, sol.t.size // 400)))
    if sol.t_events[0].size:
        return dict(t=float(sol.t_events[0][0]), status=sol.status,
                    smin=smin, clamped=op.n_clamped, tend=float(sol.t[-1]))
    return dict(t=None, status=sol.status, smin=smin,
                clamped=op.n_clamped, tend=float(sol.t[-1]))


def _cell(d):
    if d["t"] is not None:
        return f"{d['t']:.6f}"
    if d["status"] != 0:
        return f"no event; integrator stopped at t={d['tend']:.3f}"
    return f"no event; min S(1)={d['smin']:.4f} to t={d['tend']:.1f}"


# --------------------------------------------- XII (the published signature)
def t12_row(N, ramp):
    """Nozzle forcing at theta0 = +15 deg, the case of Fig. 10 of Ramos."""
    p = Parameters(Fr=10.0, We=50.0, theta0_deg=15.0, amplitude=0.1,
                   St=0.1, ramp_cycles=ramp)
    d = _degeneracy(N, p, seed_theta0=15.0)
    return [N, p.protocol, _cell(d), f"{d['smin']:.4f}", d["clamped"]]


# ------------------------------------------------- XIII (amplitude map)
def t13_row(A, Stg):
    p = Parameters(Fr=10.0, We=50.0, theta0_deg=0.0, Cpmax=1.0,
                   pressure_ratio0=1.0, amplitude=0.0, ramp_cycles=0.0,
                   body_amplitude=A, St_g=Stg)
    d = _degeneracy(32, p, tf=40.0)
    return [f"{A:.3f}", f"{Stg:.3f}", _cell(d), f"{d['smin']:.4f}",
            d["clamped"]]


# ---------------------------------- XIV (resolution across the jump in A)
def t14_row(A, N):
    p = Parameters(Fr=10.0, We=50.0, theta0_deg=0.0, Cpmax=1.0,
                   pressure_ratio0=1.0, amplitude=0.0, ramp_cycles=0.0,
                   body_amplitude=A, St_g=0.5)
    d = _degeneracy(N, p, tf=40.0)
    return [f"{A:.3f}", N, _cell(d), f"{d['smin']:.4f}", d["clamped"]]


# ------------------------------------------ XV (transversality of the tip)
def t15_row(We, theta0):
    """Smallest transversality margin h'(L) over Cpn and beta at fixed
    (We, theta0).  The proposition of Sec. V assumes h'(L) < 0; this
    records where, if anywhere, that assumption is close to failing."""
    worst = None
    for Cpn in (-0.5, -0.25, 0.0, 0.25, 0.5):
        for beta in (0.005, 0.02, 0.05, 0.1, 0.2):
            tp = ThickParameters(Fr=10.0, We=We, theta0_deg=theta0,
                                 beta=beta, Cpmax=1.0,
                                 pressure_ratio0=1.0 + Cpn)
            L, sol = reference_solve(tp, z_max=400.0)
            if not np.isfinite(L):
                continue
            dz = 1.0e-5 * max(L, 1.0)
            def h(z):
                R, u, _ = sol.sol(z)
                return R * R - 0.5 * beta / u
            hp = (h(L) - h(L - dz)) / dz
            if worst is None or hp > worst[0]:
                worst = (hp, Cpn, beta, L)
    if worst is None:
        return [f"{We:g}", f"{theta0:g}", "no solution", "", "", ""]
    hp, Cpn, beta, L = worst
    return [f"{We:g}", f"{theta0:g}", f"{hp:.6e}", f"{Cpn:.2f}",
            f"{beta:.3f}", f"{L:.6f}"]


# ------------------------------------- XVI (quasi-static gains elsewhere)
def t16_row(Fr, We, Cpn):
    """Closed-loop static gain against the St -> 0 limit of the dynamic one."""
    from static_gain import march, closed_loop
    L0, V0 = march(1.0, Cpn, Fr=Fr, We=We)
    d = 1.0e-3
    r0 = 1.0 + Cpn            # Cpmax = 1 here, so p_i(0)/p_e = 1 + Cpn
    Lp, _, Cp_p = closed_loop(1.0 + d, 1.0, V0, ratio0=r0, Fr=Fr, We=We)
    Lm, _, Cp_m = closed_loop(1.0 - d, 1.0, V0, ratio0=r0, Fr=Fr, We=We)
    gL_static = (Lp - Lm) / (2.0 * d) / L0
    gC_static = abs((Cp_p - Cp_m) / (2.0 * d))

    AMP, St, N = 0.02, 0.005, 32
    base = Parameters(Fr=Fr, We=We, theta0_deg=0.0, Cpmax=1.0,
                      pressure_ratio0=1.0 + Cpn)
    w, prob = solve_steady(N, base)
    p = Parameters(Fr=Fr, We=We, theta0_deg=0.0, Cpmax=1.0,
                   pressure_ratio0=1.0 + Cpn, amplitude=AMP, St=St,
                   ramp_cycles=2.0)
    T = p.period
    op = UnsteadySpectral(N, p, dealias=True)
    y0 = op.state_from_steady(w, prob)
    op.volume_reference = op.volume(op.unpack(y0, 0.0)[1], float(y0[-1]))
    NC = max(8, int(np.ceil(70.0 / T)) + 5)
    te = np.linspace(0.0, NC * T, 60 * NC + 1)
    sol = op.integrate(y0, NC * T, method="DOP853", rtol=1e-11, atol=1e-13,
                       t_eval=te, max_step=0.05)
    obs = [op.observables(t, sol.y[:, k]) for k, t in enumerate(sol.t)]
    tt = np.array([o["t"] for o in obs])
    LL = np.array([o["L"] for o in obs])
    CC = np.array([o["Cpn"] for o in obs])
    m = tt >= (NC - 3) * T
    wf = 2 * np.pi * St
    M = np.column_stack([np.ones(m.sum()), np.cos(wf * tt[m]),
                         np.sin(wf * tt[m])])
    cL, *_ = np.linalg.lstsq(M, LL[m], rcond=None)
    cC, *_ = np.linalg.lstsq(M, CC[m], rcond=None)
    gL_dyn = np.hypot(cL[1], cL[2]) / (AMP * cL[0])
    gC_dyn = np.hypot(cC[1], cC[2]) / AMP
    return [f"{Fr:g}", f"{We:g}", f"{Cpn:.2f}", f"{L0:.6f}",
            f"{gL_static:.6f}", f"{gL_dyn:.6f}",
            f"{abs(gL_dyn / gL_static - 1):.2e}",
            f"{gC_static:.6f}", f"{gC_dyn:.6f}",
            f"{abs(gC_dyn / gC_static - 1):.2e}"]


# ------------------------------------------- XVII (finer resolutions)
def t17_row(N):
    p = Parameters(**BODY)
    d = _degeneracy(N, p, eps=1.0e-6)
    return [N, _cell(d), d["clamped"]]


# ------------------------------------------ XVIII (finer FD meshes)
def t18_row(M):
    return t9_row(M)


# ------------------------------ XIX (sensitivity to the residual window)
def t19_row():
    """Off-grid residual against the lower edge of the evaluation window."""
    from offgrid import fine_operators, offgrid_residual
    pf = Parameters(Fr=10, We=50, theta0_deg=0, amplitude=0.1, St=0.1,
                    ramp_cycles=2.0)
    T, NCYC, N = pf.period, 9, 32
    w, prob = solve_steady(N, Parameters(Fr=10, We=50, theta0_deg=0))
    op = UnsteadySpectral(N, pf, dealias=True)
    y0 = op.state_from_steady(w, prob)
    op.volume_reference = op.volume(op.unpack(y0, 0.0)[1], float(y0[-1]))
    sol = op.integrate(y0, NCYC * T, method="DOP853", rtol=1e-10,
                       atol=1e-12, max_step=0.05)
    out = []
    for lo in (1.0e-6, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20):
        ef, P = fine_operators(op, eta_min=lo)
        out.append(f"{lo:g}:{offgrid_residual(op, sol.t[-1], sol.y[:, -1], ef, P)['max']:.2e}")
    return [N, " ".join(out)]


# --------------------------- XX (the second degeneracy, better diagnosed)
def t20_row(a, St, N):
    """Steepening of the mass-flux wave: min m and max |du/dz| together."""
    base = Parameters(Fr=10, We=50, theta0_deg=0)
    w, prob = solve_steady(N, base)
    p = Parameters(Fr=10.0, We=50.0, theta0_deg=0.0, amplitude=a, St=St,
                   ramp_cycles=2.0)
    op = UnsteadySpectral(N, p, dealias=True)
    y0 = op.state_from_steady(w, prob)
    op.volume_reference = op.volume(op.unpack(y0, 0.0)[1], float(y0[-1]))

    def ev(t, y):
        m, S, u, v, L = op.unpack(y, t)
        return float(np.max(np.abs(op.D @ u)) / L) - 50.0
    ev.terminal, ev.direction = True, +1
    sol = op.integrate(y0, 40.0, method="DOP853", rtol=1e-10, atol=1e-12,
                       max_step=0.02, events=ev)
    k = range(0, sol.t.size, max(1, sol.t.size // 400))
    st = [op.unpack(sol.y[:, i], sol.t[i]) for i in k]
    mmin = min(float(x[0].min()) for x in st)
    smin = min(float(x[1][-1]) for x in st)
    umax = max(float(np.max(np.abs(op.D @ x[2])) / x[4]) for x in st)
    tev = float(sol.t_events[0][0]) if sol.t_events[0].size else None
    return [f"{a:.3f}", f"{St:.3f}", N,
            "none" if tev is None else f"{tev:.6f}",
            f"{mmin:.4e}", f"{smin:.4f}", f"{umax:.3e}",
            f"{sol.status}", f"{sol.t[-1]:.3f}", op.n_clamped]


# --------------------------------------------------------------------
# Tables defined after the functions above must be registered here,
# not in the literal, because the literal is evaluated before those
# functions exist.
# --------------------------------------------------------------------

TABLES.update({
    "table12_signature_theta15": {
        "header": ["N", "protocol", "t_star", "min_S1", "n_clamped"],
        "tasks": [(t12_row, (N, r)) for r in (0.0, 2.0)
                  for N in (24, 32, 48, 64, 96)],
    },
    "table13_amplitude_map": {
        "header": ["A", "St_g", "t_star", "min_S1", "n_clamped"],
        "tasks": [(t13_row, (A, S)) for S in (0.250, 0.375, 0.500, 0.750,
                                              1.000)
                  for A in np.round(np.arange(0.30, 0.801, 0.05), 3)],
    },
    "table14_amplitude_resolution": {
        "header": ["A", "N", "t_star", "min_S1", "n_clamped"],
        "tasks": [(t14_row, (A, N)) for A in (0.45, 0.50, 0.55)
                  for N in (24, 32, 48, 64)],
    },
    "table15_transversality_margin": {
        "header": ["We", "theta0_deg", "worst_hprime_L", "at_Cpn",
                   "at_beta", "L"],
        "tasks": [(t15_row, (We, th))
                  for th in (-30.0, -15.0, 0.0, 15.0, 30.0)
                  for We in (2.0, 5.0, 10.0, 25.0, 50.0, 75.0, 100.0,
                             200.0)],
    },
    "table16_gains_elsewhere": {
        "header": ["Fr", "We", "Cpn", "L0", "gain_L_static",
                   "gain_L_dynamic", "rel_diff_L", "gain_Cpn_static",
                   "gain_Cpn_dynamic", "rel_diff_Cpn"],
        "tasks": [(t16_row, a) for a in ((10.0, 50.0, 0.0),
                                         (10.0, 50.0, 0.5),
                                         (10.0, 25.0, 0.0),
                                         (10.0, 100.0, 0.0),
                                         (1000.0, 50.0, 0.0),
                                         (10.0, 50.0, -0.25))],
    },
    "table17_finer_resolution": {
        "header": ["N", "t_star", "n_clamped"],
        "tasks": [(t17_row, (N,)) for N in (128, 160)],
    },
    "table18_finer_fd_mesh": {
        "header": ["M", "t_star_upwind_fd"],
        "tasks": [(t18_row, (M,)) for M in (1280, 2560)],
    },
    "table19_residual_window": {
        "header": ["N", "residual_by_window_lower_edge"],
        "tasks": [(t19_row, ())],
    },
    "table20_second_degeneracy": {
        "header": ["a", "St", "N", "t_at_uz_50", "min_m", "min_S1",
                   "max_uz", "status", "t_end", "n_clamped"],
        "tasks": [(t20_row, (a, St, N)) for a, St in ((0.10, 0.50),
                                                      (0.02, 0.50))
                  for N in (24, 32, 48, 64, 96)],
    },

})


# Flat list of (table_name, row_index, function, args).
#
# The order is fixed explicitly rather than derived, for two reasons.  The
# expensive tables come first, so that a scheduler starts the long tasks
# earliest and the wall time is set by the longest task rather than by the
# dispatch order.  And the order never changes when a table is added: new
# tables are appended, so the index of an existing task is stable and a
# partial run can be completed later without renumbering.
TABLE_ORDER = [
    "table9_cross_check",            # indices   0 -   3
    "table7_frequency_response",     #           4 -  14
    "table8_degeneracy_time",        #          15 -  29
    "table4_forced_response",        #          30 -  34
    "table1_spectral_convergence",   #          35 -  44
    "table2_ramos1992_membrane",     #          45 -  50
    "table3_near_critical",          #          51 -  58
    "table5_thickness",              #          59 -  64
    "table6_ramos1993_thickness",    #          65 -  75
    "table11_frequency_sweep",       #          76 - 155
    "table10_beta_law",              #         156 - 255
    "table12_signature_theta15",     #         256 - 265
    "table13_amplitude_map",         #         266 - 320
    "table14_amplitude_resolution",  #         321 - 332
    "table15_transversality_margin", #         333 - 372
    "table16_gains_elsewhere",       #         373 - 378
    "table17_finer_resolution",      #         379 - 380
    "table18_finer_fd_mesh",         #         381 - 382
    "table19_residual_window",       #         383
    "table20_second_degeneracy",     #         384 - 393
]
assert set(TABLE_ORDER) == set(TABLES), "TABLE_ORDER is out of step with TABLES"

TASKS = []
for name in TABLE_ORDER:
    for i, (fn, args) in enumerate(TABLES[name]["tasks"]):
        TASKS.append((name, i, fn, args))


# ------------------ XXI (is m -> 0 a mechanism or a loss of resolution?)
def t21_row(a, St, N, eps):
    """Terminal event on min m = eps, with the diagnostics that decide.

    Table XX leaves an open question: at a = 0.10 the minimum of m falls to
    about 1e-14 while the largest axial velocity gradient never exceeds
    0.67.  Along characteristics m = m0 exp(-int u_z dt), so a gradient
    bounded by 0.67 over t <= 8.4 cannot take m below exp(-5.6) = 3.7e-3.
    Either the gradient is far larger at instants the sampling misses, or
    the computed m does not solve the equation.  This task settles it: the
    event is placed on m itself, and the trajectory integral of the
    gradient is accumulated continuously rather than sampled, so the two
    sides of that inequality can be compared directly.
    """
    base = Parameters(Fr=10, We=50, theta0_deg=0)
    w, prob = solve_steady(N, base)
    p = Parameters(Fr=10.0, We=50.0, theta0_deg=0.0, amplitude=a, St=St,
                   ramp_cycles=2.0)
    op = UnsteadySpectral(N, p, dealias=True)
    y0 = op.state_from_steady(w, prob)
    op.volume_reference = op.volume(op.unpack(y0, 0.0)[1], float(y0[-1]))

    def ev(t, y):
        return float(op.unpack(y, t)[0].min()) - eps
    ev.terminal, ev.direction = True, -1
    te = np.linspace(0.0, 40.0, 40001)      # 1e-3 in time, not 400 samples
    sol = op.integrate(y0, 40.0, method="DOP853", rtol=1e-10, atol=1e-12,
                       max_step=0.01, events=ev, t_eval=te)

    uz = np.empty(sol.t.size)
    mmin = np.empty(sol.t.size)
    smin = np.empty(sol.t.size)
    for k in range(sol.t.size):
        m, S, u, v, L = op.unpack(sol.y[:, k], sol.t[k])
        uz[k] = np.max(np.abs(op.D @ u)) / L
        mmin[k] = m.min()
        smin[k] = S[-1]
    integral = float(np.trapezoid(uz, sol.t))
    k = int(np.argmin(mmin))
    m, S, u, v, L = op.unpack(sol.y[:, k], sol.t[k])
    eta_m = float(op.eta[int(np.argmin(m))])

    tev = float(sol.t_events[0][0]) if sol.t_events[0].size else None
    return [f"{a:.3f}", f"{St:.3f}", N, f"{eps:.0e}",
            "none" if tev is None else f"{tev:.6f}",
            f"{mmin.min():.4e}", f"{eta_m:.4f}", f"{smin[k]:.4f}",
            f"{uz.max():.4f}", f"{integral:.4f}",
            f"{np.exp(-integral):.3e}", f"{sol.status}",
            f"{sol.t[-1]:.4f}", op.n_clamped]


TABLES["table21_characteristic_bound"] = {
    "header": ["a", "St", "N", "eps", "t_at_m_eps", "min_m", "eta_at_min_m",
               "S1_there", "max_uz_over_run", "integral_uz_dt",
               "exp_minus_integral", "status", "t_end", "n_clamped"],
    "tasks": [(t21_row, (0.10, 0.50, N, e))
              for e in (1.0e-2, 1.0e-4, 1.0e-6, 1.0e-8)
              for N in (32, 48, 64, 96)],
}
TABLE_ORDER.append("table21_characteristic_bound")
TASKS.extend((("table21_characteristic_bound", i, fn, args))
             for i, (fn, args) in
             enumerate(TABLES["table21_characteristic_bound"]["tasks"]))
