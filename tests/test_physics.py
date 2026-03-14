"""
test_physics.py — physics-based correctness checks.

These verify that uPDE produces the right answer, not just that it runs.
They complement the snapshot tests: if a snapshot regresses, these tell you
*why* (wrong physics vs. wrong numerics vs. precision drift).

Tests
-----
  test_diffusion_steady_state   — Dirichlet heat eq converges to u = 1 - x
  test_exponential_decay        — du/dt = -λu matches exp(-λt) to 1e-12
  test_advection_conserves_mass — periodic advection preserves ∫u dx
  test_wave_energy_nongrowth    — upwind wave eq energy only dissipates

Notes on tolerances
-------------------
  Diffusion (t=10, D=0.1, n=64): L-inf < 1e-3  (slowest mode ~ exp(-D π² t))
  Decay (no diffusion, uniform): node-wise error < 1e-12
  Mass conservation: relative drift < 1e-5
  Wave energy: upwind scheme is dissipative; energy must not *grow* (< 1% growth
  allowed to account for floating-point rounding; actual drift is ~5% loss which
  is the expected numerical dissipation of the upwind stencil).
"""

import numpy as np
import pytest

from upde import PDE, PDESystem, HeatEquation, AdvectionDiffusion, WaveEquation


def test_diffusion_steady_state(grid_1d_64):
    """
    1D heat equation with Dirichlet BCs u(0)=1, u(1)=0.
    Exact steady state: u_∞ = 1 - x.  L-inf error < 1e-3 at t = 10.
    """
    x  = grid_1d_64
    eq = HeatEquation("T", x=x, diffusivity=0.1)
    eq.set_bc(side="left",  kind="dirichlet", value=1.0)
    eq.set_bc(side="right", kind="dirichlet", value=0.0)
    eq.set_ic(0.0)
    sol = eq.solve((0, 10.0), method="RK45", rtol=1e-8, atol=1e-10)
    assert sol.success

    exact = 1.0 - x
    err   = np.max(np.abs(sol.T[:, -1] - exact))
    assert err < 1e-3, f"Steady-state L-inf error {err:.2e} exceeds 1e-3"


def test_exponential_decay(grid_1d_32):
    """
    du/dt = -λu on uniform domain, Neumann BCs, uniform IC = 1.
    Exact solution: u(t) = exp(-λt)  (spatially uniform for all t).
    Node-wise error < 1e-12 at t = 1.
    """
    x   = grid_1d_32
    lam = 3.0

    eq = PDE("u", x=x)
    eq.add_source(expr=lambda x, u: -lam * u)
    eq.set_bc(side="left",  kind="neumann", value=0.0)
    eq.set_bc(side="right", kind="neumann", value=0.0)
    eq.set_ic(1.0)

    t_eval = np.linspace(0, 1.0, 101)
    sol    = eq.solve((0, 1.0), method="RK45", t_eval=t_eval,
                      rtol=1e-9, atol=1e-11)
    assert sol.success

    # The field is spatially uniform: check a single interior node.
    u_node  = sol.u[16, :]
    u_exact = np.exp(-lam * t_eval)
    max_err = np.max(np.abs(u_node - u_exact))
    assert max_err < 1e-12, f"Decay max error {max_err:.2e} exceeds 1e-12"


def test_advection_conserves_mass(grid_1d_64):
    """
    Pure advection of a Gaussian on a periodic domain.
    ∫u dx is an invariant of the PDE; relative drift must be < 1e-5.
    """
    x  = grid_1d_64
    dx = x[1] - x[0]
    ic = np.exp(-((x - 0.3) ** 2) / 0.02)

    eq = AdvectionDiffusion("phi", x=x, velocity=1.0, diffusivity=0.0)
    eq.set_bc(kind="periodic")
    eq.set_ic(ic)

    t_eval = np.linspace(0, 0.8, 17)
    sol    = eq.solve((0, 0.8), method="RK45", t_eval=t_eval,
                      rtol=1e-9, atol=1e-11)
    assert sol.success

    mass_0 = sol.phi[:, 0].sum() * dx
    for k in range(1, sol.phi.shape[1]):
        mass_k    = sol.phi[:, k].sum() * dx
        rel_drift = abs(mass_k - mass_0) / (abs(mass_0) + 1e-30)
        assert rel_drift < 1e-5, (
            f"Mass drift {rel_drift:.2e} at t={t_eval[k]:.3f} exceeds 1e-5"
        )


def test_wave_energy_nongrowth(grid_1d_64):
    """
    WaveEquation on a periodic domain with upwind advection.
    The upwind stencil is dissipative, so energy E = ½∫(uₜ² + c²uₓ²)dx
    must never *increase* beyond the initial value (< 1% growth tolerance
    for floating-point rounding). Actual behaviour is ~5% energy loss —
    the inherent numerical dissipation of the upwind scheme.
    """
    x  = grid_1d_64
    dx = x[1] - x[0]
    c  = 1.0

    ns = WaveEquation("u", "ut", x=x, speed=c)
    ns.u.set_bc(kind="periodic")
    ns.ut.set_bc(kind="periodic")
    ns.u.set_ic(np.sin(2 * np.pi * x))
    ns.ut.set_ic(np.zeros_like(x))

    t_eval = np.linspace(0, 1.0, 21)
    sol    = ns.solve((0, 1.0), t_eval=t_eval, method="RK45",
                      rtol=1e-9, atol=1e-11)
    assert sol.success

    def energy(k):
        u  = sol.u[:, k]
        ut = sol.ut[:, k]
        ux = np.gradient(u, dx)
        return 0.5 * (ut ** 2 + c ** 2 * ux ** 2).sum() * dx

    E_0 = energy(0)
    for k in range(1, len(t_eval)):
        E_k       = energy(k)
        rel_growth = (E_k - E_0) / (abs(E_0) + 1e-30)
        assert rel_growth < 0.01, (
            f"Wave energy grew by {rel_growth*100:.2f}% at t={t_eval[k]:.2f} "
            f"(upwind scheme should only dissipate)"
        )
