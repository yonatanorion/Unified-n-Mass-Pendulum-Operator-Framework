#!/usr/bin/env python3
"""Run the numerical validations reported in the contribution."""

from pathlib import Path
import sys

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from src.assembly import assemble_system
from src.modal import solve_modal
from src.nonlinear import nonlinear_terms, state_derivative, total_energy
from src.propagator import peano_baker_step, propagate_linear


CASES = [
    {
        "n": 2,
        "type": "uniform",
        "m": [1.0] * 2,
        "l": [1.0] * 2,
        "R0": [0.1, 0.05, 0.0, 0.0],
    },
    {
        "n": 3,
        "type": "uniform",
        "m": [1.0] * 3,
        "l": [1.0] * 3,
        "R0": [0.1, 0.05, 0.02, 0.0, 0.0, 0.0],
    },
    {
        "n": 3,
        "type": "non-uniform",
        "m": [2.0, 1.5, 0.8],
        "l": [1.2, 0.9, 0.6],
        "R0": [0.1, 0.05, 0.02, 0.0, 0.0, 0.0],
    },
    {
        "n": 5,
        "type": "uniform",
        "m": [1.0] * 5,
        "l": [1.0] * 5,
        "R0": [0.1, 0.05, 0.02, 0.01, 0.005, 0.0, 0.0, 0.0, 0.0, 0.0],
    },
]


def structural_scaling_table():
    print("\n" + "=" * 64)
    print("TABLE 1: Linear matrix assembly structure")
    print("=" * 64)
    print(f"{'n':<6} {'M entries':<14} {'K entries':<14} {'Total':<14}")
    print("-" * 64)

    for n in [2, 4, 6, 8, 10]:
        print(f"{n:<6} {n**2:<14} {n:<14} {n**2 + n:<14}")


def linear_validation_table():
    print("\n" + "=" * 94)
    print("TABLE 2: Maximum error in theta_1(t) over t in [0, 8] s")
    print("=" * 94)
    print(
        f"{'System':<12} {'Parameters':<14} {'dt (s)':<10} "
        f"{'Series vs expm':<20} {'Modal vs expm':<20}"
    )
    print("-" * 94)

    t = np.linspace(0.0, 8.0, 801)
    dt = 0.01

    for case in CASES:
        A, M, K = assemble_system(case["m"], case["l"])
        R0 = np.asarray(case["R0"], dtype=float)

        Phi = peano_baker_step(A, dt, N=20)
        R_series = propagate_linear(Phi, R0, t)
        R_modal = solve_modal(M, K, R0, t)
        R_expm = np.asarray([expm(A * ti) @ R0 for ti in t])

        series_error = np.max(np.abs(R_series[:, 0] - R_expm[:, 0]))
        modal_error = np.max(np.abs(R_modal[:, 0] - R_expm[:, 0]))
        print(
            f"n={case['n']:<10} {case['type']:<14} {dt:<10.2f} "
            f"{series_error:<20.2e} {modal_error:<20.2e}"
        )

        if series_error > 5e-13 or modal_error > 5e-13:
            raise AssertionError("linear validation exceeded its error tolerance")


def runge_kutta_comparison_table():
    print("\n" + "=" * 72)
    print("TABLE 3: Discrete propagator consistency for n=3 non-uniform")
    print("=" * 72)
    print(f"{'Method':<28} {'dt (s)':<12} {'Max error (rad)':<20}")
    print("-" * 72)

    masses = [2.0, 1.5, 0.8]
    lengths = [1.2, 0.9, 0.6]
    R0 = np.array([0.1, 0.05, 0.02, 0.0, 0.0, 0.0])
    A, _, _ = assemble_system(masses, lengths)

    def exact_solution(t_value):
        return expm(A * t_value) @ R0

    def rk4_step(y, dt_value):
        k1 = A @ y
        k2 = A @ (y + dt_value / 2 * k1)
        k3 = A @ (y + dt_value / 2 * k2)
        k4 = A @ (y + dt_value * k3)
        return y + dt_value / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    for method, dt in [("Scaled series (N=20)", 0.1), ("RK4", 0.1), ("RK4", 0.01)]:
        t = np.arange(0.0, 8.0 + dt / 2, dt)
        states = np.zeros((len(t), len(R0)))
        states[0] = R0

        if method.startswith("Scaled"):
            Phi = peano_baker_step(A, dt, N=20)
            states = propagate_linear(Phi, R0, t)
        else:
            for index in range(1, len(t)):
                states[index] = rk4_step(states[index - 1], dt)

        reference = np.asarray([exact_solution(ti) for ti in t])
        error = np.max(np.abs(states[:, 0] - reference[:, 0]))
        print(f"{method:<28} {dt:<12.2f} {error:<20.2e}")


def nonlinear_energy_validation():
    """Reproduce the reviewer's double-pendulum energy test."""
    print("\n" + "=" * 72)
    print("TABLE 4: Corrected nonlinear double-pendulum energy balance")
    print("=" * 72)

    masses = np.ones(2)
    lengths = np.ones(2)
    initial_state = np.array([np.deg2rad(5.0), 0.0, 0.0, 0.0])
    times = np.linspace(0.0, 6.0, 601)

    def flawed_state_derivative(_time, state, masses_value, lengths_value, g):
        """Former piecewise-sign equation, retained only as a regression check."""
        theta = state[:2]
        velocity = state[2:]
        M, _, gravity = nonlinear_terms(
            theta, velocity, masses_value, lengths_value, g
        )
        suffix_masses = np.cumsum(masses_value[::-1])[::-1]
        flawed_c = np.zeros(2)
        for d in range(2):
            for k in range(2):
                coefficient = (
                    suffix_masses[max(d, k)]
                    * lengths_value[d]
                    * lengths_value[k]
                )
                sign = 1.0 if d <= k else -1.0
                flawed_c[d] += (
                    coefficient
                    * velocity[k] ** 2
                    * np.sin(theta[d] - theta[k])
                    * sign
                )
        acceleration = np.linalg.solve(M, -(flawed_c + gravity))
        return np.concatenate((velocity, acceleration))

    drifts = {}
    for label, derivative in [
        ("Corrected equation", state_derivative),
        ("Former piecewise sign", flawed_state_derivative),
    ]:
        solution = solve_ivp(
            derivative,
            (times[0], times[-1]),
            initial_state,
            args=(masses, lengths, 9.81),
            method="DOP853",
            t_eval=times,
            rtol=1e-12,
            atol=1e-14,
            max_step=0.01,
        )
        if not solution.success:
            raise RuntimeError(solution.message)

        energy = np.asarray(
            [total_energy(state, masses, lengths) for state in solution.y.T]
        )
        drifts[label] = np.max(np.abs(energy - energy[0]))
        print(f"{label:<24}: {drifts[label]:.3e}")

    if drifts["Corrected equation"] > 1e-10:
        raise AssertionError("corrected nonlinear equations do not conserve energy")
    if drifts["Former piecewise sign"] < 1e-6:
        raise AssertionError("energy regression check did not detect the old sign error")


def main():
    structural_scaling_table()
    linear_validation_table()
    runge_kutta_comparison_table()
    nonlinear_energy_validation()
    print("\nAll validations completed successfully.")


if __name__ == "__main__":
    main()
