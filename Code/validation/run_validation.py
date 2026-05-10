#!/usr/bin/env python
"""
Run all numerical validations for the linear framework
and print tables exactly as in the manuscript.
"""

import numpy as np
import time as tm
import sys
from scipy.linalg import expm

sys.path.append('.')
from src.assembly import assemble_system
from src.propagator import propagate_linear, peano_baker_step

def symbolic_scaling_table():
    print("\n" + "="*70)
    print("TABLA 1: Symbolic generation scaling")
    print("="*70)
    print(f"{'n':<6} {'Generation time (s)':<22} {'Number of terms':<18} {'O(n²) scaling':<15}")
    print("-" * 70)
    
    for n in [2, 4, 6, 8, 10]:
        masses, lengths = [1.0]*n, [1.0]*n
        start = tm.perf_counter()
        assemble_system(masses, lengths)
        elapsed = tm.perf_counter() - start
        
        terms = 2 * n**2 + 3 * n + 1
        print(f"{n:<6} {elapsed:.4f}                 {terms:<18} {n**2:<15}")

def linear_validation_table():
    print("\n" + "="*80)
    print("TABLA 2: Maximum absolute error in θ₁(t) over t∈[0,8] s")
    print("="*80)
    print(f"{'System':<12} {'Parameters':<12} {'State dim.':<12} {'Δt (s)':<10} {'Max error (rad)':<15}")
    print("-" * 80)

    t = np.linspace(0, 8, 801)
    g = 9.81
    dt = 0.01

    cases = [
        {'n': 2, 'type': 'uniform',     'm': [1.0]*2,           'l': [1.0]*2,           'R0': [0.1, 0.05, 0.0, 0.0]},
        {'n': 3, 'type': 'uniform',     'm': [1.0]*3,           'l': [1.0]*3,           'R0': [0.1, 0.05, 0.02, 0.0, 0.0, 0.0]},
        {'n': 3, 'type': 'non-uniform', 'm': [2.0, 1.5, 0.8],   'l': [1.2, 0.9, 0.6],   'R0': [0.1, 0.05, 0.02, 0.0, 0.0, 0.0]},
        {'n': 5, 'type': 'uniform',     'm': [1.0]*5,           'l': [1.0]*5,           'R0': [0.1, 0.05, 0.02, 0.01, 0.005, 0.0, 0.0, 0.0, 0.0, 0.0]}
    ]

    for case in cases:
        A, _, _ = assemble_system(case['m'], case['l'], g)
        R0 = np.array(case['R0'])
        
        # 1. Calcular Phi
        Phi = peano_baker_step(A, dt, N=20)
        # 2. Propagar
        R_vol = propagate_linear(Phi, R0, t)
        
        R_expm = np.zeros_like(R_vol)
        for i, ti in enumerate(t):
            R_expm[i] = expm(A * ti) @ R0
            
        err = np.max(np.abs(R_vol[:, 0] - R_expm[:, 0])) 
        print(f"n={case['n']:<10} {case['type']:<12} {2*case['n']:<12} {dt:<10} {err:.2e}")

def runge_kutta_comparison_table():
    print("\n" + "="*80)
    print("TABLA 3: Comparison of Peano-Baker and RK4 for n=3 non-uniform (linear regime)")
    print("="*80)
    print(f"{'Method':<25} {'Δt (s)':<10} {'Max error (rad)':<20} {'Time (s)':<12}")
    print("-" * 80)

    g = 9.81
    masses = [2.0, 1.5, 0.8]
    lengths = [1.2, 0.9, 0.6]
    R0 = np.array([0.1, 0.05, 0.02, 0.0, 0.0, 0.0])
    A, _, _ = assemble_system(masses, lengths, g)

    def exact_solution(t_val):
        return expm(A * t_val) @ R0

    def rk4_step(A_mat, y, dt_val):
        k1 = A_mat @ y
        k2 = A_mat @ (y + dt_val/2 * k1)
        k3 = A_mat @ (y + dt_val/2 * k2)
        k4 = A_mat @ (y + dt_val * k3)
        return y + dt_val/6 * (k1 + 2*k2 + 2*k3 + k4)

    # 1. Peano-Baker dt = 0.1
    t_01 = np.linspace(0, 8, 81)
    start = tm.perf_counter()
    Phi_01 = peano_baker_step(A, 0.1, N=20)
    R_pb = propagate_linear(Phi_01, R0, t_01)
    time_pb = tm.perf_counter() - start
    err_pb = np.max(np.abs(R_pb[:, 0] - np.array([exact_solution(ti)[0] for ti in t_01])))
    print(f"{'Peano-Baker (N=20)':<25} {'0.1':<10} {err_pb:.1e}             {time_pb:.4f}")

    # 2. RK4 dt = 0.1
    start = tm.perf_counter()
    R_rk4_1 = np.zeros((len(t_01), len(R0)))
    R_rk4_1[0] = R0
    for i in range(1, len(t_01)):
        R_rk4_1[i] = rk4_step(A, R_rk4_1[i-1], 0.1)
    time_rk4_1 = tm.perf_counter() - start
    err_rk4_1 = np.max(np.abs(R_rk4_1[:, 0] - np.array([exact_solution(ti)[0] for ti in t_01])))
    print(f"{'RK4':<25} {'0.1':<10} {err_rk4_1:.1e}             {time_rk4_1:.4f}")

    # 3. RK4 dt = 0.01
    t_001 = np.linspace(0, 8, 801)
    start = tm.perf_counter()
    R_rk4_2 = np.zeros((len(t_001), len(R0)))
    R_rk4_2[0] = R0
    for i in range(1, len(t_001)):
        R_rk4_2[i] = rk4_step(A, R_rk4_2[i-1], 0.01)
    time_rk4_2 = tm.perf_counter() - start
    err_rk4_2 = np.max(np.abs(R_rk4_2[:, 0] - np.array([exact_solution(ti)[0] for ti in t_001])))
    print(f"{'RK4':<25} {'0.01':<10} {err_rk4_2:.1e}             {time_rk4_2:.4f}")

def main():
    symbolic_scaling_table()
    linear_validation_table()
    runge_kutta_comparison_table()
    print("\nSimulaciones lineales completadas exitosamente.")

if __name__ == '__main__':
    main()