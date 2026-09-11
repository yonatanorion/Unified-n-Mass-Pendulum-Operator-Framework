"""
A Unified Structural and Operator Framework for the Linear n-Mass Pendulum
"""

from .assembly import assemble_system
from .propagator import peano_baker_step, propagate_linear
from .modal import solve_modal
from .nonlinear import nonlinear_terms, state_derivative, total_energy

__version__ = "1.2.0"
__all__ = [
    "assemble_system",
    "peano_baker_step",
    "propagate_linear",
    "solve_modal",
    "nonlinear_terms",
    "state_derivative",
    "total_energy",
]