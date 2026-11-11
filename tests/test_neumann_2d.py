"""
test_neumann_2d.py — 2D Neumann and natural (no-BC) boundaries.

Same treatment as 1D (see test_neumann_1d.py): mirror ghost
phi_ghost = phi_1 + 2*d*g along each axis, with g the outward normal
derivative (array along the edge; partial-wall mask= BCs supported),
scaled by D inside the diffusion stencil.

Covers
------
  exactness     — quadratic steady solutions in x and in y, all four walls
  convergence   — 2nd order for a non-polynomial steady solution
  conservation  — zero-flux walls conserve the trapezoidal integral
  flux balance  — d/dt int T = D * sum of boundary fluxes, incl. partial
                  (mask=) Neumann walls
  sources       — act at Neumann boundary nodes
  add_term      — Div_flux_x/y match add_diffusion (Dxx/Dyy keep their
                  replicated-ghost convention, consistent with Dx/Dy)
"""

import numpy as np
import pytest

from upde import PDE, PDESystem


def _trapz2(F, x, y):
    wx = np.full(len(x), x[1] - x[0]); wx[[0, -1]] *= 0.5
    wy = np.full(len(y), y[1] - y[0]); wy[[0, -1]] *= 0.5
    return wx @ F @ wy


def _grid(nx=21, ny=11):
    return np.linspace(0.0, 1.0, nx), np.linspace(0.0, 1.0, ny)


class TestSteadyExactness:

    # (dirichlet side, neumann side, coordinate the solution depends on,
    #  sign: +1 if the Neumann wall's outward normal is +axis)
    CASES = [
        ("left",   "right", "x", +1),
        ("right",  "left",  "x", -1),
        ("bottom", "top",   "y", +1),
        ("top",    "bottom","y", -1),
    ]

    @pytest.mark.parametrize("dir_side, neu_side, axis, sgn", CASES)
    @pytest.mark.parametrize("D, g", [(1.0, 0.0), (1.0, 0.7), (0.01, -1.3)])
    def test_quadratic(self, dir_side, neu_side, axis, sgn, D, g):
        """D T'' + S = 0 along one axis; zero-flux on the other two walls."""
        x, y = _grid()
        S    = 100.0 * D
        eq   = PDE("T", x=x, y=y)
        eq.add_diffusion(diffusivity=D)
        eq.add_source(expr=S)
        others = {"x": ["bottom", "top"], "y": ["left", "right"]}[axis]
        eq.set_bc(side=others, kind="neumann", value=0.0)
        eq.set_bc(side=dir_side, kind="dirichlet", value=0.0)
        eq.set_bc(side=neu_side, kind="neumann",   value=g)
        T = eq.solve_steady().T
        s = x if axis == "x" else y
        # Map so that the Dirichlet wall is at xi = 0, Neumann wall at xi = 1
        xi = s if sgn > 0 else 1.0 - s
        exact = -S / (2 * D) * xi**2 + (g + S / D) * xi
        exact = exact[:, None] if axis == "x" else exact[None, :]
        np.testing.assert_allclose(T, np.broadcast_to(exact, T.shape),
                                   atol=1e-9 * np.abs(exact).max())

    def test_second_order_convergence(self):
        """T'' = e^x in x, zero flux in y; T(0) = 1, T'(1) = e  ->  T = e^x."""
        errs = []
        for n in (21, 41, 81):
            x, y = np.linspace(0.0, 1.0, n), np.linspace(0.0, 1.0, 7)
            eq = PDE("T", x=x, y=y)
            eq.add_diffusion(diffusivity=1.0)
            eq.add_source(expr=lambda x, y: -np.exp(x))
            eq.set_bc(side=["bottom", "top"], kind="neumann", value=0.0)
            eq.set_bc(side="left",  kind="dirichlet", value=1.0)
            eq.set_bc(side="right", kind="neumann",   value=np.e)
            T = eq.solve_steady().T
            errs.append(np.max(np.abs(T - np.exp(x)[:, None])))
        orders = np.log2(np.array(errs[:-1]) / np.array(errs[1:]))
        assert np.all(orders > 1.9), orders


class TestTransient:

    def test_zero_flux_conserves_integral(self):
        x, y = _grid(31, 25)
        eq = PDE("T", x=x, y=y)
        eq.add_diffusion(diffusivity=lambda x, y: 0.01 + 0.05 * x * y)
        eq.set_bc(side="all", kind="neumann", value=0.0)
        eq.set_ic(lambda x, y: np.exp(-40 * ((x - 0.3)**2 + (y - 0.6)**2)))
        sol = eq.solve(t_span=(0.0, 1.0), method="BDF", rtol=1e-8, atol=1e-11)
        m0 = _trapz2(sol.T[:, :, 0], x, y)
        m1 = _trapz2(sol.T[:, :, -1], x, y)
        assert abs(m1 - m0) / m0 < 1e-6

    def test_prescribed_flux_all_walls(self):
        """Neumann g on all four walls: d/dt int T = D * g * perimeter."""
        x, y = _grid(21, 17)
        D, g = 0.1, 0.5
        eq = PDE("T", x=x, y=y)
        eq.add_diffusion(diffusivity=D)
        eq.set_bc(side="all", kind="neumann", value=g)
        eq.set_ic(0.0)
        sol = eq.solve(t_span=(0.0, 1.0), rtol=1e-9, atol=1e-12)
        assert _trapz2(sol.T[:, :, -1], x, y) == pytest.approx(4 * D * g, rel=1e-6)

    def test_partial_wall_mask(self):
        """Neumann g on part of the right wall only (mask=)."""
        x, y = _grid(21, 17)
        D, g = 0.1, 0.5
        mask = np.zeros((len(x), len(y)), dtype=bool)
        mask[-1, 1 : len(y) // 2] = True     # skip the corner (ambiguous normal)
        eq = PDE("T", x=x, y=y)
        eq.add_diffusion(diffusivity=D)
        eq.set_bc(side="all", kind="neumann", value=0.0)
        eq.set_bc(mask=mask, kind="neumann", value=g)
        eq.set_ic(0.0)
        sol = eq.solve(t_span=(0.0, 1.0), rtol=1e-9, atol=1e-12)
        wy = np.full(len(y), y[1] - y[0]); wy[[0, -1]] *= 0.5
        expected = D * g * wy[mask[-1, :]].sum()
        assert _trapz2(sol.T[:, :, -1], x, y) == pytest.approx(expected, rel=1e-6)

    def test_source_acts_at_boundary_nodes(self):
        x, y = _grid(11, 9)
        eq = PDE("T", x=x, y=y)
        eq.add_diffusion(diffusivity=0.01)
        eq.add_source(expr=1.0)
        eq.set_bc(side="all", kind="neumann", value=0.0)
        eq.set_ic(0.0)
        sol = eq.solve(t_span=(0.0, 0.3), rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(sol.T[:, :, -1], 0.3, rtol=1e-8)


class TestAddTerm:

    def _rhs(self, use_term):
        x, y = _grid(15, 12)
        X, Y = np.meshgrid(x, y, indexing="ij")
        T    = np.cos(2 * X) * np.sin(3 * Y + 0.2)
        eq   = PDE("T", x=x, y=y)
        if use_term == "Div_flux":
            eq.add_term(lambda T, Div_flux_x, Div_flux_y:
                        Div_flux_x(0.3, T) + Div_flux_y(0.3, T))
        else:
            eq.add_diffusion(diffusivity=0.3)
        eq.set_bc(side="left",   kind="neumann", value=0.4)
        eq.set_bc(side="right",  kind="neumann", value=-0.2)
        eq.set_bc(side="bottom", kind="neumann", value=0.1)
        eq.set_bc(side="top",    kind="neumann", value=0.0)
        return PDESystem([eq])._rhs(0.0, T.ravel())

    def test_div_flux_matches_add_diffusion(self):
        np.testing.assert_allclose(self._rhs("Div_flux"), self._rhs(None),
                                   atol=1e-12)
