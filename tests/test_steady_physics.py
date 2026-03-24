# Last modified: 2026-03-24 14:51 UTC
"""
test_steady_physics.py
======================
Physics-based correctness tests for PDE.solve_steady() and
PDESystem.solve_steady().

These verify the right answer (exact solutions, Green's identity), not just
that the solver runs. They complement the snapshot tests in test_regression.py:
if a snapshot regresses, these tell you *why*.

Add the TestSteadyPhysics class to tests/test_physics.py,
or run this file standalone with pytest (requires conftest.py fixtures).

Tests
-----
  steady_1d_linear_exact          1D Laplace, T=1-x, error < 1e-12
  steady_1d_dirichlet_bcs         boundary nodes satisfy prescribed values
  steady_1d_poisson_exact         d2T/dx2 + pi^2 sin(pi x) = 0, O(dx^2) error
  steady_1d_nonlinear_residual    k=1+T^2, PDE residual < 1e-6
  steady_2d_laplace_exact         2D Laplace, T=1-x, machine precision
  steady_2d_poisson_exact         2D Poisson, sinusoidal source, O(dx^2) error
  steady_coupled_species_balance  A+B->C, Green's identity balance < 1e-2
  steady_returns_steady_solution  return type is SteadySolution
  steady_1d_field_shape           1D field shape is (nx,) — no time dimension
  steady_2d_field_shape           2D field shape is (nx, ny) — no time dimension
  steady_raises_for_external_field  PDE.solve_steady raises for external field refs
  steady_default_guess_accuracy   zero guess gives correct answer for linear problems
"""

import numpy as np
import pytest

from upde import PDE, PDESystem, SteadySolution


class TestSteadyPhysics:

    # ------------------------------------------------------------------
    # 1D — linear
    # ------------------------------------------------------------------

    def test_steady_1d_linear_exact(self, grid_1d_64):
        """
        1D Laplace: d2T/dx2 = 0, T(0)=1, T(1)=0.
        Exact: T = 1 - x.  Error must be at machine precision.
        """
        x  = grid_1d_64
        eq = PDE('T', x=x)
        eq.add_diffusion(diffusivity=1.0)
        eq.set_bc(side='left',  kind='dirichlet', value=1.0)
        eq.set_bc(side='right', kind='dirichlet', value=0.0)

        sol = eq.solve_steady()

        assert sol.residual < 1e-10
        np.testing.assert_allclose(
            sol.T, 1.0 - x, atol=1e-12,
            err_msg="Linear 1D steady: T must equal 1-x to near-machine precision",
        )

    def test_steady_1d_dirichlet_bcs(self, grid_1d_64):
        """Boundary nodes must exactly satisfy the prescribed Dirichlet values."""
        x  = grid_1d_64
        eq = PDE('T', x=x)
        eq.add_diffusion(diffusivity=1.0)
        eq.set_bc(side='left',  kind='dirichlet', value=2.0)
        eq.set_bc(side='right', kind='dirichlet', value=0.5)

        sol = eq.solve_steady()

        assert abs(sol.T[0]  - 2.0) < 1e-12, "Left BC not satisfied"
        assert abs(sol.T[-1] - 0.5) < 1e-12, "Right BC not satisfied"

    def test_steady_1d_poisson_exact(self, grid_1d_64):
        """
        1D Poisson: d2T/dx2 + pi^2 sin(pi x) = 0, T(0)=T(1)=0.
        Exact: T = sin(pi x).  Error must be O(dx^2).
        """
        x  = grid_1d_64
        eq = PDE('T', x=x)
        eq.add_diffusion(diffusivity=1.0)
        eq.add_source(expr=lambda x: np.pi**2 * np.sin(np.pi * x))
        eq.set_bc(side='left',  kind='dirichlet', value=0.0)
        eq.set_bc(side='right', kind='dirichlet', value=0.0)

        sol = eq.solve_steady()

        assert sol.residual < 1e-10, f"1D Poisson residual {sol.residual:.2e} not converged"
        T_exact = np.sin(np.pi * x)
        err = np.max(np.abs(sol.T - T_exact))
        assert err < 1e-3, f"1D Poisson max error {err:.2e} exceeds O(dx^2) bound"

    # ------------------------------------------------------------------
    # 1D — nonlinear
    # ------------------------------------------------------------------

    def test_steady_1d_nonlinear_residual(self, grid_1d_64):
        """
        Nonlinear conductivity k = 1 + T^2.
        No closed-form solution, but the PDE residual must be at the
        discretisation floor after convergence.
        """
        x  = grid_1d_64
        eq = PDE('T', x=x)
        eq.add_diffusion(diffusivity=lambda x, T: 1.0 + T**2)
        eq.set_bc(side='left',  kind='dirichlet', value=1.0)
        eq.set_bc(side='right', kind='dirichlet', value=0.0)

        sol = eq.solve_steady(guess=np.linspace(1.0, 0.0, len(x)))

        assert sol.success
        assert sol.residual < 1e-6, (
            f"Nonlinear 1D: residual {sol.residual:.2e} exceeds 1e-6"
        )
        assert np.all(sol.T >= -0.01) and np.all(sol.T <= 1.01), (
            "Solution left [0,1] bracket — likely wrong branch or divergence"
        )

    # ------------------------------------------------------------------
    # 2D — linear
    # ------------------------------------------------------------------

    def test_steady_2d_laplace_exact(self, grid_2d_16):
        """
        2D Laplace: T=1 left, T=0 right, Neumann top/bottom.
        Exact: T = 1 - x (uniform in y).  Machine precision.
        """
        x, y = grid_2d_16
        X, _ = np.meshgrid(x, y, indexing='ij')

        eq = PDE('T', x=x, y=y)
        eq.add_diffusion(diffusivity=1.0)
        eq.set_bc(side='left',   kind='dirichlet', value=1.0)
        eq.set_bc(side='right',  kind='dirichlet', value=0.0)
        eq.set_bc(side='bottom', kind='neumann',   value=0.0)
        eq.set_bc(side='top',    kind='neumann',   value=0.0)

        sol = eq.solve_steady()

        assert sol.residual < 1e-10
        np.testing.assert_allclose(
            sol.T, 1.0 - X, atol=1e-12,
            err_msg="2D Laplace: T must equal 1-x to near-machine precision",
        )

    def test_steady_2d_poisson_exact(self, grid_2d_16):
        """
        2D Poisson: Lap(T) + sin(pi x) sin(pi y) = 0, T=0 walls.
        Exact: T = sin(pi x) sin(pi y) / (2 pi^2).  Error O(dx^2).
        """
        x, y = grid_2d_16
        X, Y = np.meshgrid(x, y, indexing='ij')
        dx   = x[1] - x[0]

        eq = PDE('T', x=x, y=y)
        eq.add_diffusion(diffusivity=1.0)
        eq.add_source(expr=lambda x, y: np.sin(np.pi * x) * np.sin(np.pi * y))
        for side in ('left', 'right', 'bottom', 'top'):
            eq.set_bc(side=side, kind='dirichlet', value=0.0)

        sol = eq.solve_steady()

        assert sol.residual < 1e-10, f"2D Poisson residual {sol.residual:.2e} not converged"
        T_exact = np.sin(np.pi * X) * np.sin(np.pi * Y) / (2 * np.pi**2)
        err   = np.max(np.abs(sol.T - T_exact))
        bound = 10 * dx**2
        assert err < bound, (
            f"2D Poisson max error {err:.2e} exceeds O(dx^2) bound {bound:.2e}"
        )

    # ------------------------------------------------------------------
    # Coupled system
    # ------------------------------------------------------------------

    def test_steady_coupled_species_balance(self):
        """
        Steady A+B->C.  Green's identity:
          D * (dc/dx|_1 - dc/dx|_0) = integral(k cA cB)
        Error must be O(dx^2) ~ 1e-2 on a 100-point grid.
        """
        nx = 100
        x  = np.linspace(0, 1, nx)
        dx = x[1] - x[0]
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

        def grad_left(f):
            return (-3*f[0] + 4*f[1] - f[2]) / (2*dx)

        def grad_right(f):
            return (3*f[-1] - 4*f[-2] + f[-3]) / (2*dx)

        total_rxn = np.trapezoid(k * sol.cA * sol.cB, x)

        for name, field in [('cA', sol.cA), ('cB', sol.cB)]:
            lhs = abs(grad_right(field) - grad_left(field))
            err = abs(lhs - total_rxn)
            assert err < 1e-2, (
                f"Species balance error for {name}: {err:.2e} exceeds 1e-2"
            )

    # ------------------------------------------------------------------
    # API contracts
    # ------------------------------------------------------------------

    def test_steady_returns_steady_solution(self, grid_1d_32):
        """solve_steady() must return a SteadySolution, not a PDESolution."""
        x  = grid_1d_32
        eq = PDE('T', x=x)
        eq.add_diffusion(diffusivity=1.0)
        eq.set_bc(side='left',  kind='dirichlet', value=1.0)
        eq.set_bc(side='right', kind='dirichlet', value=0.0)

        sol = eq.solve_steady()
        assert isinstance(sol, SteadySolution)

    def test_steady_1d_field_shape(self, grid_1d_64):
        """1D steady field must have shape (nx,) — no time dimension."""
        x  = grid_1d_64
        eq = PDE('T', x=x)
        eq.add_diffusion(diffusivity=1.0)
        eq.set_bc(side='left',  kind='dirichlet', value=1.0)
        eq.set_bc(side='right', kind='dirichlet', value=0.0)

        sol = eq.solve_steady()
        assert sol.T.shape == (len(x),), (
            f"Expected (nx,)=({len(x)},), got {sol.T.shape}"
        )

    def test_steady_2d_field_shape(self, grid_2d_16):
        """2D steady field must have shape (nx, ny) — no time dimension."""
        x, y = grid_2d_16
        eq = PDE('T', x=x, y=y)
        eq.add_diffusion(diffusivity=1.0)
        eq.set_bc(side='left',   kind='dirichlet', value=1.0)
        eq.set_bc(side='right',  kind='dirichlet', value=0.0)
        eq.set_bc(side='bottom', kind='neumann',   value=0.0)
        eq.set_bc(side='top',    kind='neumann',   value=0.0)

        sol = eq.solve_steady()
        assert sol.T.shape == (len(x), len(y)), (
            f"Expected ({len(x)},{len(y)}), got {sol.T.shape}"
        )

    def test_steady_raises_for_external_field(self):
        """PDE.solve_steady() must raise ValueError when referencing external fields."""
        x  = np.linspace(0, 1, 32)
        eq = PDE('cA', x=x)
        eq.add_diffusion(diffusivity=1.0)
        eq.add_source(expr=lambda x, cB: -cB)   # 'cB' is an external field
        eq.set_bc(side='left',  kind='dirichlet', value=1.0)
        eq.set_bc(side='right', kind='dirichlet', value=0.0)

        with pytest.raises(ValueError, match="external"):
            eq.solve_steady()

    def test_steady_default_guess_accuracy(self, grid_1d_64):
        """
        The default zero guess must yield a correct solution for linear problems.

        Note: scipy.optimize.root (hybr) may report success=False when starting
        from an all-zero vector due to a degenerate first trust-region step.
        The residual is the authoritative convergence criterion.
        """
        x  = grid_1d_64
        eq = PDE('T', x=x)
        eq.add_diffusion(diffusivity=1.0)
        eq.set_bc(side='left',  kind='dirichlet', value=1.0)
        eq.set_bc(side='right', kind='dirichlet', value=0.0)

        sol = eq.solve_steady()   # no guess= argument

        assert sol.residual < 1e-10, (
            f"Default-guess residual {sol.residual:.2e} should be < 1e-10"
        )
        np.testing.assert_allclose(
            sol.T, 1.0 - x, atol=1e-12,
            err_msg="Default-guess solution differs from exact T=1-x",
        )
