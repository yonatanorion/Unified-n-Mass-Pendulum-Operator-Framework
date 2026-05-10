"""
A Unified Structural and Operator Framework for the Linear n-Mass Pendulum
"""

from .assembly import assemble_system
from .propagator import peano_baker_step

__version__ = "1.0.0"
__all__ = [
    "assemble_system",
    "get_matrices",
    "compute_term_count",
    "peano_baker_step",
    "estimate_error_bound",
    "reference_solution",
]