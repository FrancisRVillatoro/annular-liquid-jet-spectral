"""Finite-time loss of transversality of the free boundary.

The tip is defined by R(t, L(t)) = 0.  The formulation with a scalar L(t)
is valid while that zero is transversal, R_z(L) != 0, equivalently
S(1) = -L R_z(L) > 0.  If S(1) -> 0 then dL/dt -> infinity and the
formulation ceases to be valid.

The event time is located by an event S(1) = eps and verified under
reduction of eps, under the integration tolerance, under the resolution and
under a change of integrator.  Reading t* off the instant at which the
integration aborts would confound three different things: the event, the
tolerance of the integrator, and the start-up protocol.

The principal case is body-force forcing (Ramos, Arch. Appl. Mech. 65
(1995), Eq. 9), because the initial field is an exact equilibrium and the
forcing is C-infinity in time, so t* does not depend on the start-up
protocol.  Nozzle forcing is kept as a secondary check, and its dependence
on the protocol is shown explicitly.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import sys

import numpy as np

from annular_spectral import Parameters, UnsteadySpectral, solve_steady
from offgrid import fine_operators, offgrid_residual, modal_tail

FR, WE = 10.0, 50.0


def setup(p, N, dealias=True):
    w, prob = solve_steady(N, Parameters(Fr=p.Fr, We=p.We,
                                         theta0_deg=p.theta0_deg,
                                         Cpmax=p.Cpmax,
                                         pressure_ratio0=p.pressure_ratio0))
    op = UnsteadySpectral(N, p, dealias=dealias)
    y0 = op.state_from_steady(w, prob)
    op.volume_reference = op.volume(op.unpack(y0, 0.0)[1], float(y0[-1]))
    return op, y0


def event_time(p, N, eps, key="S", method="DOP853", rtol=1.0e-10,
               atol=1.0e-12, tf=60.0, max_step=0.02):
    """Time at which S(1), or min m, falls below eps.  None if it does not."""
    op, y0 = setup(p, N)
    if key == "S":
        def ev(t, y):
            return float(op.unpack(y, t)[1][-1]) - eps
    else:
        def ev(t, y):
            return float(op.unpack(y, t)[0].min()) - eps
    ev.terminal = True
    ev.direction = -1
    try:
        sol = op.integrate(y0, tf, method=method, rtol=rtol, atol=atol,
                           max_step=max_step, events=ev)
    except FloatingPointError:
        return None, None, None
    te = sol.t_events[0]
    if not te.size:
        return None, float(sol.t[-1]), sol.status
    return float(te[0]), float(sol.y[-1, -1]), sol.status


def body(A=0.5, Stg=0.5):
    return Parameters(Fr=FR, We=WE, theta0_deg=0.0, Cpmax=1.0,
                      pressure_ratio0=1.0, amplitude=0.0, ramp_cycles=0.0,
                      body_amplitude=A, St_g=Stg)


def nozzle(a, ramp):
    return Parameters(Fr=FR, We=WE, theta0_deg=0.0, amplitude=a, St=0.1,
                      ramp_cycles=ramp)


def section(name):
    print("\n" + "=" * 74)
    print(name)
    print("=" * 74)


which = sys.argv[1] if len(sys.argv) > 1 else "all"

# ---------------------------------------------------------------- A
if which in ("all", "A"):
    p = body()
    section(f"A.  Caso principal: {p.protocol},  A=0.5, St_g=0.5, Fr=10, We=50")

    print("\nA1. Reduction of the event threshold  (N=32, DOP853, rtol=1e-10)")
    print(f"{'eps':>10}{'t*':>18}")
    for eps in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7):
        t, _, _ = event_time(p, 32, eps)
        cell = f"{t:18.9f}" if t is not None else f"{'no alcanzado':>18}"
        print(f"{eps:10.0e}{cell}")

    print("\nA2. Convergencia en N  (eps=1e-6, DOP853, rtol=1e-10)")
    print(f"{'N':>5}{'t*':>18}{'L(t*)':>14}")
    for N in (16, 20, 24, 32, 48, 64, 96):
        t, L, _ = event_time(p, N, 1e-6)
        if t is None:
            print(f"{N:5d}{'no detecta el suceso':>18}")
        else:
            print(f"{N:5d}{t:18.9f}{L:14.6f}")

    print("\nA3. Tolerancia e integrador  (N=32, eps=1e-6)")
    print(f"{'metodo':>10}{'rtol':>10}{'t*':>18}")
    for method, rt, at in (("DOP853", 1e-8, 1e-10), ("DOP853", 1e-10, 1e-12),
                           ("DOP853", 1e-12, 1e-14), ("Radau", 1e-9, 1e-11)):
        t, _, _ = event_time(p, 32, 1e-6, method=method, rtol=rt, atol=at)
        cell = f"{t:18.9f}" if t is not None else f"{'no alcanzado':>18}"
        print(f"{method:>10}{rt:10.0e}{cell}")

    print("\nA4. Off-grid residual and spectral tail at t* - 0.02")
    print(f"{'N':>5}{'|res|off':>13}{'cola modal':>13}{'S(1)':>12}")
    TS = 10.766826
    for N in (24, 32, 48, 64):
        op, y0 = setup(p, N)
        sol = op.integrate(y0, TS - 0.02, method="DOP853", rtol=1e-11,
                           atol=1e-13, max_step=0.02)
        ef, P = fine_operators(op)
        r = offgrid_residual(op, sol.t[-1], sol.y[:, -1], ef, P)["max"]
        tl = max(modal_tail(op, sol.y[:, -1], sol.t[-1]).values())
        S1 = op.unpack(sol.y[:, -1], sol.t[-1])[1][-1]
        print(f"{N:5d}{r:13.2e}{tl:13.2e}{S1:12.4f}")

# ---------------------------------------------------------------- B
if which in ("all", "B"):
    section("B.  Secondary check: nozzle forcing")
    print("The same event, but t* depends on the start-up protocol.")
    print(f"\n{'a':>7}{'protocolo':>22}{'N':>5}{'t*':>14}")
    for a in (0.25, 0.5):
        for ramp in (0.0, 2.0):
            p = nozzle(a, ramp)
            for N in (24, 32, 48):
                t, _, _ = event_time(p, N, 1e-4, tf=60.0)
                s = f"{t:14.4f}" if t is not None else f"{'aborta':>14}"
                print(f"{a:7.2f}{p.protocol:>22}{N:5d}{s}")
    print("\nWith 'historical_transient' the integration aborts on refinement")
    print("(a Gibbs signature); with 'periodic_orbit' the event is stable in N,")
    print("but t* then corresponds to a start-up that is not the one used in the")
    print("historical computations.  Hence the body-force case is the principal one.")

# ---------------------------------------------------------------- C
if which in ("all", "C"):
    section("C.  A second and different degeneracy: min m -> 0 at high Strouhal number")
    print("Along characteristics dm/dt = -m u_z, so m -> 0 in finite time requires")
    print("u_z -> +infinity: it is a steepening of the mass-flux wave, not a")
    print("failure of the tip.  S(1) remains O(1) at the event.")
    print(f"\n{'a':>7}{'St':>7}{'N':>5}{'delta':>9}{'t*':>14}{'S(1) en t*':>13}")
    for a, St in ((0.1, 0.5), (0.02, 0.5)):
        for N in (24, 32, 48, 64):
            p = Parameters(Fr=FR, We=WE, theta0_deg=0.0, amplitude=a, St=St,
                           ramp_cycles=2.0)
            t, _, _ = event_time(p, N, 1e-8, key="m", tf=40.0)
            if t is None:
                print(f"{a:7.2f}{St:7.2f}{N:5d}{1e-8:9.0e}{'no detecta':>14}")
                continue
            op, y0 = setup(p, N)
            s2 = op.integrate(y0, t - 1e-3, method="DOP853", rtol=1e-10,
                              atol=1e-12, max_step=0.02)
            S1 = op.unpack(s2.y[:, -1], s2.t[-1])[1][-1]
            print(f"{a:7.2f}{St:7.2f}{N:5d}{1e-8:9.0e}{t:14.6f}{S1:13.4f}")
