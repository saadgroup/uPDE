# uPDE — method-of-lines PDE solver
# https://doi.org/10.1016/j.softx.2026.102715

__version__ = "1.0.0"
__author__  = "Tony Saad"
__email__   = "tony.saad@utah.edu"
__license__ = "MIT"

# Core engine
from .upde import (
    PDE,
    PDESystem,
    PDEUnsteadySolution,
    PDESteadySolution,
)

# Pre-built equation factories
from .equations import (
    NamedPDESystem,
    HeatEquation,
    AdvectionDiffusion,
    Burgers,
    ConservationLaw,
    ReactionDiffusion,
    WaveEquation,
    GrayScott,
    NavierStokes2D,
    MixtureFraction,
)

__all__ = [
    # Core
    "PDE",
    "PDESystem",
    "PDEUnsteadySolution",
    "PDESteadySolution",
    # Factories
    "NamedPDESystem",
    "HeatEquation",
    "AdvectionDiffusion",
    "Burgers",
    "ConservationLaw",
    "ReactionDiffusion",
    "WaveEquation",
    "GrayScott",
    "NavierStokes2D",
    "MixtureFraction",
]
