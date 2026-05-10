#!/usr/bin/env python
"""
Generate figures for the linear framework manuscript.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from scipy.linalg import expm

sys.path.append('.')
from src.assembly import assemble_system
from src.propagator import propagate_linear, peano_baker_step

os.makedirs('figures', exist_ok=True)

plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
})

def fig_linear_validation():
    """Generate linear validation figure for n=2, 3, 5 (Table 2 cases)."""
    t = np.linspace(0, 8, 801)
    dt = 0.01
    g = 9.81
    
    cases = [
        {'n': 2, 'type': 'uniform',     'm': [1.0]*2,           'l': [1.0]*2,           'R0': [0.1, 0.05, 0.0, 0.0]},
        {'n': 3, 'type': 'uniform',     'm': [1.0]*3,           'l': [1.0]*3,           'R0': [0.1, 0.08, 0.05, 0.0, 0.0, 0.0]},
        {'n': 3, 'type': 'non-uniform', 'm': [1.0, 0.5, 0.2],   'l': [1.0, 0.8, 0.5],   'R0': [0.1, 0.08, 0.05, 0.0, 0.0, 0.0]},
        {'n': 5, 'type': 'uniform',     'm': [1.0]*5,           'l': [1.0]*5,           'R0': [0.1, 0.08, 0.05, 0.03, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0]}
    ]
    
    fig, axes = plt.subplots(4, 2, figsize=(10, 12))
    
    for idx, case in enumerate(cases):
        A, _, _ = assemble_system(case['m'], case['l'], g)
        R0 = np.array(case['R0'])
        
        # 1. Calcular Phi
        Phi = peano_baker_step(A, dt, N=20)
        # 2. Propagar
        R_vol = propagate_linear(Phi, R0, t)
        
        R_ref = np.zeros_like(R_vol)
        for i, ti in enumerate(t):
            R_ref[i] = expm(A * ti) @ R0
            
        error = np.abs(R_vol[:, 0] - R_ref[:, 0])
        
        title_prefix = f"n={case['n']} ({case['type']})"
        
        # Trajectory Plot
        axes[idx, 0].plot(t, R_ref[:, 0], 'k-', lw=2, label='Matrix Exp (Ref)', alpha=0.7)
        axes[idx, 0].plot(t, R_vol[:, 0], 'r--', lw=1.5, label='Peano-Baker')
        axes[idx, 0].set_ylabel('θ₁ (rad)')
        axes[idx, 0].set_title(f"{title_prefix}: Linear Trajectory")
        axes[idx, 0].legend()
        axes[idx, 0].grid(True, alpha=0.3)
        
        # Error Plot
        axes[idx, 1].semilogy(t, error, 'b-', lw=1.5)
        axes[idx, 1].set_ylabel('|Δθ₁| (rad)')
        axes[idx, 1].set_title(f"{title_prefix}: Absolute Algorithmic Error")
        axes[idx, 1].grid(True, alpha=0.3)
        
        if idx == 3:
            axes[idx, 0].set_xlabel('t (s)')
            axes[idx, 1].set_xlabel('t (s)')

    plt.tight_layout()
    plt.savefig('figures/linear_validation.png', bbox_inches='tight')
    plt.close(fig)
    print("Generated figures/linear_validation.png")

def main():
    print("Generating linear validation figure...")
    fig_linear_validation()
    print("Done.")

if __name__ == '__main__':
    main()