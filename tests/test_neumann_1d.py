"""
test_neumann_1d.py — 1D Neumann and natural (no-BC) boundaries.

The 1D diffusion stencil imposes a Neumann BC through a mirror ghost node
phi_{-1} = phi_1 + 2*dx*g (g = outward normal derivative), giving the
half-cell balance at the boundary node.  All other terms (advection,
sources) are kept at the boundary node.

Covers
------
  exactness     — steady quadratic solutions are reproduced to round-off,
                  for Neumann on either side (sign convention: outward)
  convergence   — 2nd order for a non-polynomial steady solution
  diffusivity   — boundary-node RHS scales with D
  conservation  — zero-flux walls conserve the trapezoidal integral,
                  including variable D
  other terms   — sources and advection act at Neumann boundary nodes
  natural BC    — a side with no BC is zero-flux (no wrap-around)
  add_term      — Div_flux_x applies the same Neumann ghost
"""

import numpy as np
import pytest

from upde import PDE, PDESystem


def _trapz(f, x):
    return np.sum(0.5 * (f[1:] + f[:-1]) * np.diff(x))


class TestSteadyExactness:

    @pytest.mark.parametrize("g", [0.0, 0.7, -1.3])
    def test_right_neumann_quadratic(self, g):
        """D T'' + S = 0, T(0) = 0, T'(1) = g  ->  quadratic, exact on grid."""
        D, S = 0.01, 1.0
        x  = np.linspace(0.0, 1.0, 21)
        eq = PDE("T", x=x)
        eq.add_diffusion(diffusivity=D)
        eq.add_source(expr=S)
        eq.set_bc(side="left",  kind="dirichlet", value=0.0)
        eq.set_bc(side="right", kind="neumann",   value=g)
        sol = eq.solve_steady()
        exact = -S / (2 * D) * x**2 + (g + S / D) * x
        np.testing.assert_allclose(sol.T, exact, atol=1e-9 * np.abs(exact).max())

    @pytest.mark.parametrize("g", [0.0, 0.7, -1.3])
    def test_left_neumann_outward_sign(self, g):
        """Left wall: outward normal is -x, so g = -T'(0)."""
        D, S = 0.01, 1.0
        x  = np.linspace(0.0, 1.0, 21)
        eq = PDE("T", x=x)
        eq.add_diffusion(diffusivity=D)
        eq.add_source(expr=S)
        eq.set_bc(side="left",  kind="neumann",   value=g)
        eq.set_bc(side="right", kind="dirichlet", value=0.0)
        sol = eq.solve_steady()
        # T = -S/(2D) x^2 + b x + c with T'(0) = b = -g and T(1) = 0
        b = -g
        c = S / (2 * D) - b
        exact = -S / (2 * D) * x**2 + b * x + c
        np.testing.assert_allclose(sol.T, exact, atol=1e-9 * np.abs(exact).max())

    def test_second_order_convergence(self):
        """T'' = e^x, T(0) = 1, T'(1) = e  ->  T = e^x."""
        errs = []
        for n in (21, 41, 81):
            x  = np.linspace(0.0, 1.0, n)
            eq = PDE("T", x=x)
            eq.add_diffusion(diffusivity=1.0)
            eq.add_source(expr=lambda x: -np.exp(x))
            eq.set_bc(side="left",  kind="dirichlet", value=1.0)
            eq.set_bc(side="right", kind="neumann",   value=np.e)
            errs.append(np.max(np.abs(eq.solve_steady().T - np.exp(x))))
        orders = np.log2(np.array(errs[:-1]) / np.array(errs[1:]))
        assert np.all(orders > 1.9), orders


class TestTransient:

    @pytest.mark.parametrize("D", [0.01, 1.0])
    def test_boundary_rhs_scales_with_D(self, D):
        x  = np.linspace(0.0, 1.0, 51)
        dx = x[1] - x[0]
        eq = PDE("T", x=x)
        eq.add_diffusion(diffusivity=D)
        eq.set_bc(side="right", kind="neumann", value=0.0)
        T  = 1.0 - x**2
        r  = PDESystem([eq])._rhs(0.0, T)
        assert r[-1] == pytest.approx(2 * D * (T[-2] - T[-1]) / dx**2)

    def test_zero_flux_conserves_integral(self):
        x  = np.linspace(0.0, 1.0, 81)
        eq = PDE("T", x=x)
        eq.add_diffusion(diffusivity=lambda x: 0.01 + 0.05 * x)
        eq.set_bc(side="all", kind="neumann", value=0.0)
        eq.set_ic(lambda x: np.exp(-50 * (x - 0.3) ** 2))
        sol = eq.solve(t_span=(0.0, 2.0), method="BDF", rtol=1e-8, atol=1e-11)
        m0, m1 = _trapz(sol.T[:, 0], x), _trapz(sol.T[:, -1], x)
        assert abs(m1 - m0) / m0 < 1e-6

    def test_prescribed_flux_adds_mass(self):
        """d/dt int T = D*(g_left + g_right) for Neumann on both walls."""
        x, D, g = np.linspace(0.0, 1.0, 41), 0.1, 0.5
        eq = PDE("T", x=x)
        eq.add_diffusion(diffusivity=D)
        eq.set_bc(side="left",  kind="neumann", value=g)
        eq.set_bc(side="right", kind="neumann", value=g)
        eq.set_ic(0.0)
        sol = eq.solve(t_span=(0.0, 1.0), rtol=1e-9, atol=1e-12)
        assert _trapz(sol.T[:, -1], x) == pytest.approx(2 * D * g * 1.0, rel=1e-6)

    def test_source_acts_at_boundary_nodes(self):
        x  = np.linspace(0.0, 1.0, 21)
        eq = PDE("T", x=x)
        eq.add_diffusion(diffusivity=0.01)
        eq.add_source(expr=1.0)
        eq.set_bc(side="all", kind="neumann", value=0.0)
        eq.set_ic(0.0)
        sol = eq.solve(t_span=(0.0, 0.3), rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(sol.T[:, -1], 0.3, rtol=1e-8)

    def test_advection_at_neumann_outflow(self):
        x  = np.linspace(0.0, 1.0, 51)
        eq = PDE("u", x=x)
        eq.add_advection(velocity=1.0)
        eq.set_bc(side="right", kind="neumann", value=0.0)
        r = PDESystem([eq])._rhs(0.0, x.copy())   # u = x  ->  -c u_x = -1
        np.testing.assert_allclose(r[1:], -1.0)


class TestNaturalBoundary:

    def test_unset_side_is_zero_flux(self):
        """Left T = 1, right unset: steady state is T = 1 (no wrap to x=0)."""
        x  = np.linspace(0.0, 1.0, 31)
        eq = PDE("T", x=x)
        eq.add_diffusion(diffusivity=1.0)
        eq.set_bc(side="left", kind="dirichlet", value=1.0)
        np.testing.assert_allclose(eq.solve_steady().T, 1.0, atol=1e-10)

    def test_unset_matches_zero_neumann(self):
        x  = np.linspace(0.0, 1.0, 31)
        T  = np.sin(3 * x)
        rs = []
        for neumann in (False, True):
            eq = PDE("T", x=x)
            eq.add_diffusion(diffusivity=0.2)
            eq.set_bc(side="left", kind="dirichlet", value=0.0)
            if neumann:
                eq.set_bc(side="right", kind="neumann", value=0.0)
            rs.append(PDESystem([eq])._rhs(0.0, T))
        np.testing.assert_allclose(rs[0], rs[1])


class TestAddTerm:

    def test_div_flux_matches_add_diffusion(self):
        x  = np.linspace(0.0, 1.0, 31)
        T  = np.cos(2 * x)
        rs = []
        for use_term in (False, True):
            eq = PDE("T", x=x)
            if use_term:
                eq.add_term(lambda T, Div_flux_x: Div_flux_x(0.3, T))
            else:
                eq.add_diffusion(diffusivity=0.3)
            eq.set_bc(side="left",  kind="neumann", value=0.4)
            eq.set_bc(side="right", kind="neumann", value=-0.2)
            rs.append(PDESystem([eq])._rhs(0.0, T))
        np.testing.assert_allclose(rs[1], rs[0])
