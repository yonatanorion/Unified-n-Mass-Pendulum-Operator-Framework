"""
Figures for the AJP version of the paper.

Fig. 2  (modes.png)
  (a) First four mode shapes of a uniform 40-mass chain, as horizontal
      displacements of the masses, compared with the continuous hanging chain
      of effective length L + l/2 whose free end lies l/2 below the lowest
      mass (z = height above the lowest mass; both normalized there).
  (b) Relative error of the three lowest frequencies versus the hanging-chain
      values omega_k = (j_{0,k}/2) sqrt(g/L), with the empirical 1/(4n) line
      (discrete frequencies are lower, as for a chain lengthened by l/2).

Fig. 3  (nonlinear.png)
  (a) Wall time of one exact nonlinear acceleration evaluation: dense
      M(theta) + LU versus the tridiagonal tension system.
  (b) Deviation of the nonlinear solution of a uniform 50-mass chain from the
      linear modal solution, relative to the initial amplitude eps.

Usage: python Code/validation/figures_ajp.py [--outdir Paper/figures]
"""

import os

# Single-threaded linear algebra so that the timings in Fig. 3(a) are reproducible.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
from scipy.integrate import solve_ivp  # noqa: E402
from scipy.special import j0, jn_zeros  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src import chain as C  # noqa: E402
from src.structure import modal_matrix, natural_frequencies, suffix_masses  # noqa: E402

G = 9.81
BLUE, RED, GREY = "#1F5E8C", "#B5462F", "#555555"


def dense_acceleration(theta, omega, m, l, g=G):
    S = suffix_masses(m)
    idx = np.arange(m.size)
    Smax = S[np.maximum.outer(idx, idx)]
    dth = theta[:, None] - theta[None, :]
    M = Smax * np.outer(l, l) * np.cos(dth)
    c = (Smax * np.outer(l, l) * np.sin(dth)) @ (omega**2)
    return np.linalg.solve(M, -(c + S * g * l * np.sin(theta)))


def best_time(fn, rep):
    b = np.inf
    for _ in range(rep):
        t0 = time.perf_counter(); fn(); b = min(b, time.perf_counter() - t0)
    return b


def fig_modes(plt, out):
    n, L = 40, 1.0
    m, l = np.full(n, 1.0 / n), np.full(n, L / n)
    omega, U = modal_matrix(m, l)
    z_mass = np.clip(L - np.cumsum(l), 0.0, None)  # height of each mass above the free end
    h = 0.5 * L / n                                # half a link: effective continuum length L + h
    zc = np.linspace(0, L, 400)
    jz = jn_zeros(0, 4)
    fig, ax = plt.subplots(1, 2, figsize=(9.0, 4.1), constrained_layout=True)
    for k in range(4):
        x = np.cumsum(l * U[:, k])                 # horizontal displacements x = E L theta
        x = x / x[-1]                              # normalize at the free end
        u = j0(jz[k] * np.sqrt((zc + h) / (L + h))) / j0(jz[k] * np.sqrt(h / (L + h)))
        off = 2.6 * k
        ax[0].plot(u + off, zc, color=GREY, lw=1.0)
        ax[0].plot(x + off, z_mass, "o", ms=2.6, color=BLUE)
        ax[0].axvline(off, color="k", lw=0.4, alpha=0.4)
        ax[0].text(off, 1.04, f"$k={k+1}$", ha="center", fontsize=9)
    ax[0].set_ylim(-0.05, 1.12); ax[0].set_xticks([])
    ax[0].set_ylabel("height above the lowest mass, $z/L$")
    ax[0].set_title("(a) Modes: $n=40$ beads vs continuum", fontsize=10)
    ax[0].plot([], [], "o", color=BLUE, ms=3, label="$n=40$ chain (horizontal displacement)")
    ax[0].plot([], [], color=GREY, label="hanging chain of length $L+l/2$")
    ax[0].legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=1)

    exact = jn_zeros(0, 3) / 2 * np.sqrt(G / L)
    ns = np.array([4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096])
    err = np.array([np.abs(natural_frequencies(np.full(k, 1.0 / k), np.full(k, L / k),
                                                select="i", select_range=(0, 2)) - exact) / exact for k in ns])
    for k, mk in zip(range(3), ["o", "s", "^"]):
        ax[1].loglog(ns, err[:, k], mk + "-", ms=4, lw=1, color=[BLUE, "#2E9E6A", RED][k], label=f"mode {k+1}")
    ax[1].loglog(ns, 1 / (4 * ns), "k--", lw=0.8, label="$1/(4n)$")
    ax[1].set_xlabel("number of masses $n$"); ax[1].set_ylabel("relative frequency error")
    ax[1].set_title("(b) Convergence to the hanging chain", fontsize=10)
    ax[1].grid(True, which="both", alpha=0.3); ax[1].legend(frameon=False, fontsize=8)
    fig.savefig(out, dpi=200)
    return err, ns


def fig_nonlinear(plt, out):
    rng = np.random.default_rng(3)
    rows = []
    for n in [3, 10, 30, 100, 300, 1000, 3000]:
        m, l = rng.uniform(0.5, 2, n), rng.uniform(0.5, 2, n) / n
        th, om = rng.uniform(-1, 1, n), rng.normal(0, 1, n)
        rep = 7 if n <= 1000 else 2
        rows.append((n, best_time(lambda: dense_acceleration(th, om, m, l), rep),
                     best_time(lambda: C.angular_acceleration(th, om, m, l), rep),
                     np.abs(C.angular_acceleration(th, om, m, l) - dense_acceleration(th, om, m, l)).max()
                     / np.abs(dense_acceleration(th, om, m, l)).max()))
    n_s = 50
    m, l = np.ones(n_s) / n_s, np.ones(n_s) / n_s
    omega, U = modal_matrix(m, l)
    shape = U[:, 0] / np.abs(U[:, 0]).max()
    t = np.linspace(0, 5.0, 201)
    small = []
    for eps in [0.2, 0.1, 0.05, 0.025, 0.0125]:
        th0 = eps * shape
        s = solve_ivp(C.state_derivative, (0, 5.0), np.r_[th0, np.zeros(n_s)], args=(m, l),
                      method="DOP853", rtol=1e-12, atol=1e-14, t_eval=t)
        lin = np.outer(th0, np.cos(omega[0] * t))
        small.append((eps, np.abs(s.y[:n_s] - lin).max() / eps))
    fig, ax = plt.subplots(1, 2, figsize=(9.0, 3.6), constrained_layout=True)
    n = np.array([r[0] for r in rows], float)
    ax[0].loglog(n, [r[1] for r in rows], "o-", color=RED, ms=4, label="dense $\\mathbf{M}(\\boldsymbol{\\theta})$ + LU")
    ax[0].loglog(n, [r[2] for r in rows], "s-", color=BLUE, ms=4, label="tension system")
    ax[0].set_xlabel("number of masses $n$"); ax[0].set_ylabel("time per evaluation (s)")
    ax[0].set_title("(a) Exact nonlinear accelerations", fontsize=10)
    ax[0].grid(True, which="both", alpha=0.3); ax[0].legend(frameon=False, fontsize=8)
    eps = np.array([r[0] for r in small]); es = np.array([r[1] for r in small])
    ax[1].loglog(eps, es, "o-", color=BLUE, ms=4, label="nonlinear vs linear, $n=50$")
    ax[1].loglog(eps, es[-1] * (eps / eps[-1]) ** 2, "k--", lw=0.8, label="$\\propto\\varepsilon^2$")
    import matplotlib.ticker
    ax[1].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax[1].set_xticks(eps); ax[1].set_xticklabels([f"{e:g}" for e in eps])
    ax[1].set_xlabel("initial amplitude $\\varepsilon$ (rad)"); ax[1].set_ylabel("max deviation $/\\ \\varepsilon$")
    ax[1].set_title("(b) Small-amplitude limit", fontsize=10)
    ax[1].grid(True, which="both", alpha=0.3); ax[1].legend(frameon=False, fontsize=8)
    fig.savefig(out, dpi=200)
    return rows, small


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=os.path.join(os.path.dirname(__file__), "..", "..", "Paper", "figures"))
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9})
    os.makedirs(a.outdir, exist_ok=True)
    err, ns = fig_modes(plt, os.path.join(a.outdir, "modes.png"))
    print("Relative frequency error vs hanging chain (modes 1-3):")
    for k, e in zip(ns, err):
        print(f"  n={k:>5d}  " + "  ".join(f"{x:.3e}" for x in e) + f"   1/(4n)={1/(4*k):.3e}")
    rows, small = fig_nonlinear(plt, os.path.join(a.outdir, "nonlinear.png"))
    print("\nExact nonlinear acceleration: n, dense (s), tension (s), rel. difference")
    for r in rows:
        print(f"  {r[0]:>5d}  {r[1]:.2e}  {r[2]:.2e}  {r[3]:.1e}")
    print("\nSmall-amplitude deviation / eps:")
    for e, d in small:
        print(f"  eps={e:<7g} {d:.3e}")
    p = np.polyfit(np.log([s[0] for s in small]), np.log([s[1] for s in small]), 1)[0]
    print(f"  order {p:.2f}")


if __name__ == "__main__":
    main()
