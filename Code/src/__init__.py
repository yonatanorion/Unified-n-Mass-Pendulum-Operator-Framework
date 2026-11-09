"""
The multiple pendulum as a beaded string: closed-form tridiagonal structure,
normal modes and exact O(n) nonlinear dynamics of the planar n-mass pendulum.
"""

from .assembly import assemble_system
from .propagator import peano_baker_step, propagate_linear
from .modal import solve_modal
from .nonlinear import nonlinear_terms, state_derivative, total_energy
from .structure import (
    suffix_masses,
    inverse_mass_tridiagonal,
    dynamic_matrix_tridiagonal,
    apply_dynamic_matrix,
    jacobi_matrix,
    natural_frequencies,
    modal_matrix,
)
from .symplectic import linear_energy, max_frequency, verlet_propagate
from . import chain

__version__ = "3.1.0"
__all__ = [
    "assemble_system",
    "peano_baker_step",
    "propagate_linear",
    "solve_modal",
    "nonlinear_terms",
    "state_derivative",
    "total_energy",
    "suffix_masses",
    "inverse_mass_tridiagonal",
    "dynamic_matrix_tridiagonal",
    "apply_dynamic_matrix",
    "jacobi_matrix",
    "natural_frequencies",
    "modal_matrix",
    "linear_energy",
    "max_frequency",
    "verlet_propagate",
    "chain",
]