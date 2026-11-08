"""
Exact nonlinear dynamics in O(n) via the tridiagonal tension system.

(a) Cost of one evaluation of the exact angular accelerations versus n:
    dense M(theta) assembly + LU solve (O(n^3)) vs tension system (O(n)).
(b) Long-time energy error for a chaotic triple pendulum released from
    horizontal: RATTLE (symplectic), RK4 and adaptive DOP853, all using the
    O(n) right-hand side.
(c) Second-order convergence of RATTLE against a DOP853 reference (tol 1e-13).
(d) Small-amplitude limit: the nonlinear solution approaches the linear one
    with relative difference O(eps^2) for initial amplitude eps (n = 50).

Usage:  python Code/validation/benchmark_nonlinear.py [--quick] [--out FIG.png]
"""

import argparse
import os
import sys
import time

import numpy as np
from scipy.integrate import solve_ivp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src import chain as C  # noqa: E402
from src.structure import modal_matrix, suffix_masses  # noqa: E402

G = 9.81
M3, L3 = np.array([2.0, 1.5, 0.8]), np.array([1.2, 0.9, 0.6])


def dense_acceleration(theta, omega, m, l, g=G):
    """Vectorized dense reference: M(theta) thddot = -(c + g_vec)."""
    S = suffix_masses(m)
    idx = np.arange(m.size)
    Smax = S[np.maximum.outer(idx, idx)]
    dth = theta[:, None] - theta[None, :]
    M = Smax * np.outer(l, l) * np.cos(dth)
    c = (Smax * np.outer(l, l) * np.sin(dth)) @ (omega**2)
    gv = S * g * l * np.sin(theta)
    return np.linalg.solve(M, -(c + gv))


def best_time(fn, rep):
    b = np.inf
    for _ in range(rep):
        t0 = time.perf_counter(); fn(); b = min(b, time.perf_counter() - t0)
    return b


def cost_study(quick):
    rng = np.random.default_rng(3)
    rows = []
    for n in [10, 30, 100, 300, 1000, 3000] + ([10000, 100000] if not quick else [10000]):
        m, l = rng.uniform(0.5, 2, n), rng.uniform(0.5, 2, n) / n
        th, om = rng.uniform(-1, 1, n), rng.normal(0, 1, n)
        rep = 5 if n <= 1000 else 2
        t_on = best_time(lambda: C.angular_acceleration(th, om, m, l), rep)
        t_d = best_time(lambda: dense_acceleration(th, om, m, l), rep) if n <= 3000 else np.nan
        diff = np.nan
        if n <= 3000:
            a1, a2 = C.angular_acceleration(th, om, m, l), dense_acceleration(th, om, m, l)
            diff = np.abs(a1 - a2).max() / np.abs(a2).max()
        rows.append((n, t_d, t_on, diff))
    return rows


def rk4(f, y, h, N, every):
    out = [y.copy()]
    for k in range(N):
        k1 = f(y); k2 = f(y + 0.5 * h * k1); k3 = f(y + 0.5 * h * k2); k4 = f(y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        if (k + 1) % every == 0:
            out.append(y.copy())
    return np.array(out)


def energy_study(t_end, dt=1e-3):
    th0, om0 = np.full(3, np.pi / 2), np.zeros(3)
    E0 = C.energy(th0, om0, M3, L3)
    scale = M3.sum() * G * L3.sum()
    N = int(round(t_end / dt)); every = max(1, N // 1000)
    res = {}
    t0 = time.perf_counter()
    T, TH, OM, it = C.rattle_propagate(th0, om0, M3, L3, dt, N, save_every=every)
    res[f"RATTLE, dt={dt:g}"] = (T, np.abs([C.energy(a, b, M3, L3) - E0 for a, b in zip(TH, OM)]) / scale,
                                 time.perf_counter() - t0, f"{it / N:.2f} Newton it./step")
    t0 = time.perf_counter()
    Y = rk4(lambda y: C.state_derivative(0, y, M3, L3), np.r_[th0, om0], dt, N, every)
    res[f"RK4, dt={dt:g}"] = (np.arange(len(Y)) * every * dt,
                              np.abs([C.energy(y[:3], y[3:], M3, L3) - E0 for y in Y]) / scale,
                              time.perf_counter() - t0, f"{4 * N} RHS")
    for tol in (1e-6, 1e-9):
        t0 = time.perf_counter()
        s = solve_ivp(C.state_derivative, (0, t_end), np.r_[th0, om0], args=(M3, L3), method="DOP853",
                      rtol=tol, atol=tol, t_eval=np.linspace(0, t_end, 1001))
        res[f"DOP853, tol={tol:g}"] = (s.t, np.abs([C.energy(y[:3], y[3:], M3, L3) - E0 for y in s.y.T]) / scale,
                                       time.perf_counter() - t0, f"{s.nfev} RHS")
    return res


def convergence_study(t_end=2.0):
    th0, om0 = np.full(3, np.pi / 2), np.zeros(3)
    ref = solve_ivp(C.state_derivative, (0, t_end), np.r_[th0, om0], args=(M3, L3), method="DOP853",
                    rtol=1e-13, atol=1e-13).y[:3, -1]
    rows = []
    for dt in [4e-3, 2e-3, 1e-3, 5e-4, 2.5e-4]:
        N = int(round(t_end / dt))
        _, TH, _, _ = C.rattle_propagate(th0, om0, M3, L3, dt, N, save_every=N)
        rows.append((dt, np.abs(np.angle(np.exp(1j * (TH[-1] - ref)))).max()))
    return rows


def small_amplitude_study(n=50, t_end=5.0):
    m, l = np.ones(n) / n, np.ones(n) / n
    omega, U = modal_matrix(m, l)
    shape = U[:, 0] / np.abs(U[:, 0]).max()          # fundamental mode shape
    t = np.linspace(0, t_end, 201)
    rows = []
    for eps in [0.2, 0.1, 0.05, 0.025, 0.0125]:
        th0 = eps * shape
        s = solve_ivp(C.state_derivative, (0, t_end), np.r_[th0, np.zeros(n)], args=(m, l), method="DOP853",
                      rtol=1e-12, atol=1e-14, t_eval=t)
        lin = np.outer(th0, np.ones_like(t)) * np.cos(omega[0] * t)[None, :]
        rows.append((eps, np.abs(s.y[:n] - lin).max() / eps))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "figures", "nonlinear.png"))
    args = ap.parse_args()

    cost = cost_study(args.quick)
    print("=" * 74)
    print("Cost (s) of one exact nonlinear acceleration evaluation")
    print("=" * 74)
    print(f"{'n':>8} {'dense M(theta)':>16} {'tension O(n)':>14} {'rel. difference':>17}")
    for n, td, to, df in cost:
        f = lambda x, w: f"{x:>{w}.2e}" if np.isfinite(x) else f"{'--':>{w}}"
        print(f"{n:>8d} {f(td, 16)} {f(to, 14)} {f(df, 17)}")

    t_end = 50.0 if args.quick else 200.0
    en = energy_study(t_end)
    print("\n" + "=" * 96)
    print(f"Chaotic triple pendulum (released horizontal), {t_end:g} s: energy error / (m_tot g L_tot)")
    print("=" * 96)
    for k, (T, E, wall, info) in en.items():
        q = max(1, len(E) // 10)
        print(f"{k:<20} max {E.max():.2e}  first10% {E[:q].max():.2e}  last10% {E[-q:].max():.2e}  "
              f"wall {wall:6.1f} s  ({info})")

    conv = convergence_study()
    print("\nRATTLE convergence at t = 2 s (chaotic triple pendulum):")
    for dt, e in conv:
        print(f"  dt = {dt:<8g} error = {e:.3e}")
    p = np.polyfit(np.log([r[0] for r in conv]), np.log([r[1] for r in conv]), 1)[0]
    print(f"  observed order: {p:.2f}")

    small = small_amplitude_study()
    print("\nSmall-amplitude limit (n = 50, fundamental mode): max|nonlinear - linear| / eps")
    for eps, e in small:
        print(f"  eps = {eps:<7g} {e:.3e}")
    p2 = np.polyfit(np.log([r[0] for r in small]), np.log([r[1] for r in small]), 1)[0]
    print(f"  observed order: {p2:.2f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker
    plt.rcParams.update({"font.size": 9})
    fig, ax = plt.subplots(2, 2, figsize=(10, 7.2), constrained_layout=True)
    a = ax.ravel()
    n = np.array([r[0] for r in cost], float)
    td = np.array([r[1] for r in cost]); to = np.array([r[2] for r in cost])
    ok = np.isfinite(td)
    a[0].loglog(n[ok], td[ok], "o-", color="#B5462F", label="dense $\\mathbf{M}(\\boldsymbol{\\theta})$ + LU")
    a[0].loglog(n, to, "s-", color="#1F5E8C", label="tension system, $\\mathcal{O}(n)$")
    a[0].set_xlabel("$n$"); a[0].set_ylabel("wall time per evaluation (s)")
    a[0].set_title("(a) Exact nonlinear accelerations"); a[0].grid(True, which="both", alpha=0.3); a[0].legend(frameon=False)
    for (k, (T, E, _, _)), c in zip(en.items(), ["#1F5E8C", "#2E9E6A", "#D08C2E", "#B5462F"]):
        msk = T > 0
        a[1].semilogy(T[msk], np.maximum(E[msk], 1e-17), lw=0.8, color=c, label=k)
    a[1].set_xlabel("$t$ (s)"); a[1].set_ylabel("$|E-E_0|/(m_{\\rm tot} g L_{\\rm tot})$")
    a[1].set_title("(b) Chaotic triple pendulum: energy"); a[1].grid(True, which="both", alpha=0.3)
    a[1].legend(frameon=False, fontsize=8, loc="lower right")
    dts = np.array([r[0] for r in conv]); er = np.array([r[1] for r in conv])
    a[2].loglog(dts, er, "o-", color="#1F5E8C", label="RATTLE")
    a[2].loglog(dts, er[-1] * (dts / dts[-1]) ** 2, "k--", lw=0.8, label="$\\propto\\Delta t^2$")
    a[2].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    a[2].set_xlabel("$\\Delta t$ (s)"); a[2].set_ylabel("max angle error at $t=2$ s (rad)")
    a[2].set_title("(c) RATTLE convergence (chaotic regime)"); a[2].grid(True, which="both", alpha=0.3); a[2].legend(frameon=False)
    eps = np.array([r[0] for r in small]); es = np.array([r[1] for r in small])
    a[3].loglog(eps, es, "o-", color="#1F5E8C", label="nonlinear vs linear, $n=50$")
    a[3].loglog(eps, es[-1] * (eps / eps[-1]) ** 2, "k--", lw=0.8, label="$\\propto\\varepsilon^2$")
    a[3].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    a[3].set_xlabel("initial amplitude $\\varepsilon$ (rad)"); a[3].set_ylabel("max relative deviation")
    a[3].set_title("(d) Small-amplitude limit"); a[3].grid(True, which="both", alpha=0.3); a[3].legend(frameon=False)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=200)
    print(f"\nFigure written to {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
