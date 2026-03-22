"""
upde — A lightweight Python library for solving PDEs via the method of lines.

Core classes
------------
    PDE         — descriptor for a single field equation
    PDESystem   — couples equations, validates, and solves
    PDESolution — solution object with per-field array access

Pre-packaged equations
----------------------
Single-field (return PDE):
    HeatEquation, AdvectionDiffusion, Burgers,
    ConservationLaw, ReactionDiffusion

Multi-field (return NamedPDESystem):
    WaveEquation, GrayScott, NavierStokes2D

Example
-------
    import numpy as np
    from upde import AdvectionDiffusion, PDESystem

    x  = np.linspace(0, 1, 256)
    eq = AdvectionDiffusion('T', x=x, velocity=1.0, diffusivity=0.01)
    eq.set_bc(kind='periodic')
    eq.set_ic(np.sin(2 * np.pi * x))
    sol = PDESystem([eq]).solve(t_span=(0, 1), method='RK45')
    # sol.T  shape (256, nt)
"""

from .upde import PDE, PDESystem, PDESolution


from .equations import (
    # Single-field
    HeatEquation,
    AdvectionDiffusion,
    Burgers,
    ConservationLaw,
    ReactionDiffusion,
    # Multi-field
    WaveEquation,
    GrayScott,
    NavierStokes2D,
    # chemistry is a submodule — users import as: from upde.chemistry import FlameletTable
    MixtureFraction,
    # Base class (for isinstance checks / subclassing)
    NamedPDESystem,
)

__version__ = "0.1.0"
__author__  = "Tony Saad"
__email__   = "tony.saad@utah.edu"

__all__ = [
    # Core
    "PDE", "PDESystem", "PDESolution",
    # Single-field equations
    "HeatEquation", "AdvectionDiffusion", "Burgers",
    "ConservationLaw", "ReactionDiffusion",
    # Multi-field equations
    "WaveEquation", "GrayScott", "NavierStokes2D",
    "NamedPDESystem",
]