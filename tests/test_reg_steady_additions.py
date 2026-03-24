# Last modified: 2026-03-24 14:51 UTC
"""
test_reg_steady_additions.py
============================
Golden-file (snapshot) regression tests for solve_steady().

Append the four test functions below to tests/test_regression.py,
or run this file standalone with pytest.

Problems
--------
  reg_steady_1d_linear     1D Laplace, D=1, T(0)=1 T(1)=0
  reg_steady_1d_nonlinear  1D nonlinear k=1+T^2 conduction
  reg_steady_2d_laplace    2D Laplace, T=1 left T=0 right, Neumann top/bottom
  reg_steady_coupled       Coupled A+B->C steady state (all three fields)
"""

import numpy as np
import pytest

from upde import PDE, PDESystem


def test_reg_steady_1d_linear(snapshot, grid_1d_64):
    """Golden file: linear 1D steady conduction."""
    x  = grid_1d_64
    eq = PDE('T', x=x)
    eq.add_diffusion(diffusivity=1.0)
    eq.set_bc(side='left',  kind='dirichlet', value=1.0)
    eq.set_bc(side='right', kind='dirichlet', value=0.0)

    sol = eq.solve_steady()
    assert sol.residual < 1e-10
    snapshot.check("reg_steady_1d_linear", {"T": sol.T})


def test_reg_steady_1d_nonlinear(snapshot, grid_1d_64):
    """Golden file: nonlinear 1D steady conduction, k = 1 + T^2."""
    x  = grid_1d_64
    eq = PDE('T', x=x)
    eq.add_diffusion(diffusivity=lambda x, T: 1.0 + T**2)
    eq.set_bc(side='left',  kind='dirichlet', value=1.0)
    eq.set_bc(side='right', kind='dirichlet', value=0.0)

    sol = eq.solve_steady(guess=np.linspace(1.0, 0.0, len(x)))
    assert sol.success
    snapshot.check("reg_steady_1d_nonlinear", {"T": sol.T})


def test_reg_steady_2d_laplace(snapshot, grid_2d_16):
    """Golden file: 2D Laplace, Dirichlet left/right, Neumann top/bottom."""
    x, y = grid_2d_16
    eq = PDE('T', x=x, y=y)
    eq.add_diffusion(diffusivity=1.0)
    eq.set_bc(side='left',   kind='dirichlet', value=1.0)
    eq.set_bc(side='right',  kind='dirichlet', value=0.0)
    eq.set_bc(side='bottom', kind='neumann',   value=0.0)
    eq.set_bc(side='top',    kind='neumann',   value=0.0)

    sol = eq.solve_steady()
    assert sol.residual < 1e-10
    snapshot.check("reg_steady_2d_laplace", {"T": sol.T})


def test_reg_steady_coupled(snapshot):
    """Golden file: coupled A+B->C steady-state system, all three fields."""
    nx = 64
    x  = np.linspace(0, 1, nx)
    k  = 10.0

    eqA = PDE('cA', x=x)
    eqA.add_diffusion(diffusivity=1.0)
    eqA.add_source(expr=lambda x, cA, cB: -k * cA * cB)
    eqA.set_bc(side='left',  kind='dirichlet', value=5.0)
    eqA.set_bc(side='right', kind='dirichlet', value=0.0)

    eqB = PDE('cB', x=x)
    eqB.add_diffusion(diffusivity=1.0)
    eqB.add_source(expr=lambda x, cA, cB: -k * cA * cB)
    eqB.set_bc(side='left',  kind='dirichlet', value=0.0)
    eqB.set_bc(side='right', kind='dirichlet', value=5.0)

    eqC = PDE('cC', x=x)
    eqC.add_diffusion(diffusivity=1.0)
    eqC.add_source(expr=lambda x, cA, cB: +k * cA * cB)
    eqC.set_bc(side='left',  kind='dirichlet', value=0.0)
    eqC.set_bc(side='right', kind='dirichlet', value=0.0)

    sol = PDESystem([eqA, eqB, eqC]).solve_steady(
        guess={'cA': np.linspace(5, 0, nx),
               'cB': np.linspace(0, 5, nx),
               'cC': np.zeros(nx)},
    )
    assert sol.success
    snapshot.check("reg_steady_coupled", {
        "cA": sol.cA,
        "cB": sol.cB,
        "cC": sol.cC,
    })
