"""
test_regression.py — golden-file (snapshot) regression tests.

Each test runs a well-defined PDE problem and compares the result against a
stored .npz file in tests/snapshots/.  The snapshots are committed to the repo
and serve as the gold standard.

Golden-file workflow
--------------------
  First run    → snapshots are created in tests/snapshots/, test passes
  Normal run   → result is diffed against snapshot; mismatch = FAIL
  After a deliberate change:
      pytest --rebless          # regenerates all snapshots
      git diff tests/snapshots/ # review changed fields before committing

Problems
--------
  1D
    reg_diffusion_1d       heat eq, Dirichlet, converges toward steady state
    reg_advection_1d       upwind advection, periodic Gaussian pulse
    reg_source_1d          uniform exponential decay (no diffusion)
    reg_coupled_1d         two-field A+B reaction-diffusion system
    reg_time_source_1d     source that shuts off at t=0.05 (t-injection)
    reg_time_advection_1d  advection velocity flips sign at t=0.5 (t-injection)

  2D
    reg_diffusion_2d       heat eq on 16×16, Dirichlet left/right

  Factories
    reg_burgers            Burgers viscous shock
    reg_wave               WaveEquation factory, 1D periodic sinusoidal IC
"""

import numpy as np
import pytest

from upde import PDE, PDESystem, HeatEquation, AdvectionDiffusion, Burgers, WaveEquation, MixtureFraction
from upde.chemistry import FlameletTable

# ---------------------------------------------------------------------------
# 1-D regression tests
# ---------------------------------------------------------------------------

def test_reg_diffusion_1d(snapshot, grid_1d_32):
    x  = grid_1d_32
    eq = HeatEquation("T", x=x, diffusivity=0.1)
    eq.set_bc(side="left",  kind="dirichlet", value=1.0)
    eq.set_bc(side="right", kind="dirichlet", value=0.0)
    eq.set_ic(0.0)
    sol = eq.solve((0, 2.0), method="RK45", rtol=1e-8, atol=1e-10)
    assert sol.success
    snapshot.check("reg_diffusion_1d", {"T_final": sol.T[:, -1]})


def test_reg_advection_1d(snapshot, grid_1d_64):
    x  = grid_1d_64
    ic = np.exp(-((x - 0.3) ** 2) / 0.01)
    eq = AdvectionDiffusion("phi", x=x, velocity=1.0, diffusivity=0.0)
    eq.set_bc(kind="periodic")
    eq.set_ic(ic)
    t_eval = np.linspace(0, 0.5, 6)
    sol = eq.solve((0, 0.5), method="RK45", t_eval=t_eval,
                   rtol=1e-8, atol=1e-10)
    assert sol.success
    snapshot.check("reg_advection_1d", {
        "phi_t0":   sol.phi[:, 0],
        "phi_tfin": sol.phi[:, -1],
    })


def test_reg_source_1d(snapshot, grid_1d_32):
    """Spatially uniform exponential decay: u(t) = exp(-λt)."""
    x   = grid_1d_32
    lam = 2.0
    eq  = PDE("u", x=x)
    eq.add_source(expr=lambda x, u: -lam * u)
    eq.set_bc(side="left",  kind="neumann", value=0.0)
    eq.set_bc(side="right", kind="neumann", value=0.0)
    eq.set_ic(1.0)
    t_eval = np.linspace(0, 1.0, 11)
    sol = eq.solve((0, 1.0), method="RK45", t_eval=t_eval,
                   rtol=1e-8, atol=1e-10)
    assert sol.success
    snapshot.check("reg_source_1d", {"u_all": sol.u})


def test_reg_coupled_1d(snapshot):
    """Two-field A+B reaction-diffusion system."""
    x  = np.linspace(0, 1, 32)
    k1 = 5.0

    eqA = PDE("cA", x=x)
    eqA.add_diffusion(diffusivity=0.01)
    eqA.add_source(expr=lambda x, cA, cB: -k1 * cA * cB)
    eqA.set_bc(side="left",  kind="dirichlet", value=1.0)
    eqA.set_bc(side="right", kind="dirichlet", value=0.0)
    eqA.set_ic(0.0)

    eqB = PDE("cB", x=x)
    eqB.add_diffusion(diffusivity=0.01)
    eqB.add_source(expr=lambda x, cA, cB: -k1 * cA * cB)
    eqB.set_bc(side="left",  kind="dirichlet", value=0.0)
    eqB.set_bc(side="right", kind="dirichlet", value=1.0)
    eqB.set_ic(0.0)

    sol = PDESystem([eqA, eqB]).solve(
        (0, 0.2), method="RK45", rtol=1e-8, atol=1e-10,
    )
    assert sol.success
    snapshot.check("reg_coupled_1d", {
        "cA_final": sol.cA[:, -1],
        "cB_final": sol.cB[:, -1],
    })


def test_reg_time_source_1d(snapshot, grid_1d_32):
    """Source active for t < 0.05, zero after — tests t injection in sources."""
    x = grid_1d_32

    def src(x, t):
        return np.where(t < 0.05, np.ones_like(x), np.zeros_like(x))

    eq = PDE("T", x=x)
    eq.add_source(expr=src)
    eq.set_bc(side="left",  kind="neumann", value=0.0)
    eq.set_bc(side="right", kind="neumann", value=0.0)
    eq.set_ic(0.0)
    t_eval = np.linspace(0, 0.2, 21)
    sol = eq.solve((0, 0.2), method="RK45", t_eval=t_eval,
                   rtol=1e-8, atol=1e-10)
    assert sol.success
    snapshot.check("reg_time_source_1d", {"T_all": sol.T})


def test_reg_time_advection_1d(snapshot):
    """Advection velocity flips sign at t=0.5 — tests t injection in advection."""
    x  = np.linspace(0, 2 * np.pi, 64)
    ic = np.exp(-((x - np.pi) ** 2) / 0.5)

    eq = PDE("phi", x=x)
    eq.add_advection(
        velocity=lambda x, t: np.where(t < 0.5, 1.0, -1.0) * np.ones_like(x)
    )
    eq.set_bc(kind="periodic")
    eq.set_ic(ic)
    t_eval = [0.0, 0.3, 0.7, 1.0]
    sol = eq.solve((0, 1.0), method="RK45", t_eval=t_eval,
                   rtol=1e-8, atol=1e-10)
    assert sol.success
    snapshot.check("reg_time_advection_1d", {
        "phi_t03": sol.phi[:, 1],
        "phi_t10": sol.phi[:, 3],
    })


# ---------------------------------------------------------------------------
# 2-D regression tests
# ---------------------------------------------------------------------------

def test_reg_diffusion_2d(snapshot, grid_2d_16):
    """2D heat equation: Dirichlet left=1 right=0, Neumann top/bottom."""
    x, y = grid_2d_16
    eq   = HeatEquation("T", x=x, y=y, diffusivity=0.1)
    eq.set_bc(side="left",   kind="dirichlet", value=1.0)
    eq.set_bc(side="right",  kind="dirichlet", value=0.0)
    eq.set_bc(side="bottom", kind="neumann",   value=0.0)
    eq.set_bc(side="top",    kind="neumann",   value=0.0)
    eq.set_ic(0.0)
    sol = eq.solve((0, 1.0), method="RK45", rtol=1e-8, atol=1e-10)
    assert sol.success
    snapshot.check("reg_diffusion_2d", {"T_final": sol.T[:, :, -1]})


# ---------------------------------------------------------------------------
# Factory regression tests
# ---------------------------------------------------------------------------

def test_reg_burgers(snapshot, grid_1d_64):
    """Burgers equation — viscous shock."""
    x  = grid_1d_64
    eq = Burgers("u", x=x, viscosity=0.01)
    eq.set_bc(side="left",  kind="dirichlet", value= 1.0)
    eq.set_bc(side="right", kind="dirichlet", value=-1.0)
    eq.set_ic(lambda x: np.where(x < 0.5, 1.0, -1.0))
    sol = eq.solve((0, 0.5), method="RK45", rtol=1e-8, atol=1e-10)
    assert sol.success
    snapshot.check("reg_burgers", {"u_final": sol.u[:, -1]})


def test_reg_wave(snapshot, grid_1d_64):
    """WaveEquation factory — periodic sinusoidal IC."""
    x  = grid_1d_64
    ns = WaveEquation("u", "ut", x=x, speed=1.0)
    ns.u.set_bc(kind="periodic")
    ns.ut.set_bc(kind="periodic")
    ns.u.set_ic(np.sin(2 * np.pi * x))
    ns.ut.set_ic(np.zeros_like(x))
    t_eval = np.linspace(0, 1.0, 11)
    sol = ns.solve((0, 1.0), t_eval=t_eval, method="RK45",
                   rtol=1e-8, atol=1e-10)
    assert sol.success
    snapshot.check("reg_wave", {
        "u_t0":   sol.u[:, 0],
        "u_tfin": sol.u[:, -1],
    })

# ─────────────────────────────────────────────────────────────────────────────
# Chemistry / MixtureFraction regression (golden-file) tests
# Add these three functions to tests/test_regression.py
# ─────────────────────────────────────────────────────────────────────────────

def test_reg_flamelet_burke_schumann(snapshot):
    """Golden-file test: Burke-Schumann T and Y_k profiles across Z ∈ [0, 1]."""
    from upde.chemistry import FlameletTable
    table = FlameletTable.burke_schumann(
        Z_st=0.055, T_fuel=300.0, T_ox=300.0, T_ad=2230.0
    )
    Z = np.linspace(0, 1, 100)
    snapshot.check("reg_flamelet_bs", {
        "T":     table.T(Z),
        "Y_CH4": table.Y('CH4', Z),
        "Y_O2":  table.Y('O2',  Z),
        "Y_CO2": table.Y('CO2', Z),
        "Y_H2O": table.Y('H2O', Z),
        "Y_N2":  table.Y('N2',  Z),
    })


def test_reg_mixture_fraction_1d(snapshot):
    """Golden-file test: 1-D mixture-fraction transient diffusion (step IC)."""
    nx  = 64
    x   = np.linspace(0, 1, nx)
    eq  = MixtureFraction('Z', x=x, diffusivity=1e-3)
    eq.set_bc(side='left',  kind='dirichlet', value=0.0)
    eq.set_bc(side='right', kind='dirichlet', value=1.0)
    eq.set_ic(lambda x: (x > 0.5).astype(float))
    t_eval = np.linspace(0, 0.5, 6)
    sol = PDESystem([eq]).solve(
        t_span=(0, 0.5), t_eval=t_eval, method='BDF', rtol=1e-8, atol=1e-10
    )
    assert sol.success
    snapshot.check("reg_mixture_fraction_1d", {
        "Z_t0":   sol.Z[:, 0],
        "Z_tmid": sol.Z[:, 3],
        "Z_tfin": sol.Z[:, -1],
    })


def test_reg_mixture_fraction_1d_advection(snapshot):
    """Golden-file test: 1-D mixture fraction with mean convection (u=0.05)."""
    nx = 64
    x  = np.linspace(0, 1, nx)
    eq = MixtureFraction('Z', x=x, velocity=0.05, diffusivity=1e-3)
    eq.set_bc(side='left',  kind='dirichlet', value=0.0)
    eq.set_bc(side='right', kind='dirichlet', value=1.0)
    eq.set_ic(lambda x: x)
    sol = PDESystem([eq]).solve(
        t_span=(0, 2.0), method='BDF', rtol=1e-8, atol=1e-10
    )
    assert sol.success
    snapshot.check("reg_mixture_fraction_1d_advection", {
        "Z_final": sol.Z[:, -1],
    })