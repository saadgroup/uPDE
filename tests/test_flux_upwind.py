"""
test_flux_upwind.py — first-order add_flux(scheme='upwind').

The face flux is local Lax-Friedrichs (Rusanov) on the nodal states:
one flux per face, so it is conservative where dF/du changes sign and
spreads sonic expansion fans; for a linear flux it is exact donor-cell.

Covers
------
  linear flux   — identical to add_advection(scheme='upwind'), 1D and 2D
  conservation  — periodic Burgers and traffic flow (dF/du changes sign)
  entropy       — Burgers sonic expansion fan converges with refinement
                  (no stationary expansion shock)
  shock speed   — Rankine-Hugoniot speed for a one-signed shock
"""

import numpy as np
import pytest

from upde import PDE, PDESystem, Burgers, ConservationLaw


@pytest.mark.parametrize("c", [1.3, -0.8])
def test_linear_flux_matches_advection_1d(c):
    x   = np.linspace(0.0, 1.0, 30)
    phi = np.random.default_rng(0).standard_normal(30)
    rs  = []
    for flux in (False, True):
        eq = PDE("u", x=x)
        if flux:
            eq.add_flux(flux=lambda u: c * u)
        else:
            eq.add_advection(velocity=c)
        eq.set_bc(kind="periodic")
        rs.append(PDESystem([eq])._rhs(0.0, phi))
    np.testing.assert_allclose(rs[1], rs[0], rtol=1e-6, atol=1e-8)


def test_linear_flux_matches_advection_2d():
    x, y = np.linspace(0.0, 1.0, 12), np.linspace(0.0, 1.0, 9)
    phi  = np.random.default_rng(1).standard_normal(12 * 9)
    rs   = []
    for flux in (False, True):
        eq = PDE("u", x=x, y=y)
        if flux:
            eq.add_flux(flux_x=lambda u: 0.9 * u, flux_y=lambda u: -1.2 * u)
        else:
            eq.add_advection(velocity_x=0.9, velocity_y=-1.2)
        eq.set_bc(kind="periodic", side="all")
        rs.append(PDESystem([eq])._rhs(0.0, phi))
    np.testing.assert_allclose(rs[1], rs[0], rtol=1e-6, atol=1e-8)


def test_burgers_periodic_conserves_mass():
    """u0 = 0.5 + sin x: dF/du = u changes sign; donor-cell lost 47%."""
    x  = np.linspace(0.0, 2 * np.pi, 200, endpoint=False)
    eq = Burgers("u", x=x)
    eq.set_bc(kind="periodic")
    eq.set_ic(0.5 + np.sin(x))
    sol = eq.solve(t_span=(0.0, 3.0), rtol=1e-8, atol=1e-10)
    m = sol.u.sum(axis=0)
    assert abs(m[-1] - m[0]) / abs(m[0]) < 1e-10


def test_traffic_periodic_conserves_mass():
    x  = np.linspace(0.0, 1.0, 200, endpoint=False)
    eq = ConservationLaw("r", x=x, flux=lambda r: r * (1 - r))
    eq.set_bc(kind="periodic")
    eq.set_ic(np.where((x > 0.3) & (x < 0.6), 0.9, 0.2))
    sol = eq.solve(t_span=(0.0, 0.5), rtol=1e-8, atol=1e-10)
    m = sol.r.sum(axis=0)
    assert abs(m[-1] - m[0]) / m[0] < 1e-10


def test_burgers_sonic_expansion_converges():
    """uL=-1, uR=1: exact u = x/t in |x| < t; donor-cell froze the jump."""
    T, errs = 0.5, []
    for n in (101, 401):
        x  = np.linspace(-1.0, 1.0, n)
        eq = Burgers("u", x=x)
        eq.set_bc(side="left",  kind="dirichlet", value=-1.0)
        eq.set_bc(side="right", kind="dirichlet", value=1.0)
        eq.set_ic(lambda x: np.where(x < 0.0, -1.0, 1.0))
        sol = eq.solve(t_span=(0.0, T), rtol=1e-6, atol=1e-9)
        errs.append(np.mean(np.abs(sol.u[:, -1] - np.clip(x / T, -1.0, 1.0))))
    assert errs[1] < 0.02
    assert errs[0] / errs[1] > 2.0


def test_burgers_shock_speed():
    """uL=2, uR=0: shock speed (uL+uR)/2 = 1."""
    x  = np.linspace(0.0, 4.0, 401)
    eq = Burgers("u", x=x)
    eq.set_bc(side="left", kind="dirichlet", value=2.0)
    eq.set_ic(lambda x: np.where(x < 1.0, 2.0, 0.0))
    sol = eq.solve(t_span=(0.0, 1.0), rtol=1e-8, atol=1e-10)
    u = sol.u[:, -1]
    assert x[np.argmin(np.abs(u - 1.0))] == pytest.approx(2.0, abs=0.02)
