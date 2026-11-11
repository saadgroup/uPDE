"""
test_limiters.py — limited (TVD) MUSCL schemes for add_advection / add_flux.

Covers
------
  _limited_slope      — limiter functions psi(r), zero slope at extrema
  TVD                 — no new extrema on a periodic square wave
  accuracy            — limited schemes beat first-order upwind; ~2nd order
                        convergence on a smooth profile
  direction           — c < 0 mirrors c > 0
  flux form           — linear flux reproduces add_advection; Burgers
                        sonic expansion fan has no expansion shock
  sparsity            — widened pattern for limited schemes, periodic wrap;
                        the pattern covers the true (numerical) Jacobian in
                        1D and 2D for every scheme
  steady              — 'auto' detects the limiter nonlinearity
  2D                  — TVD, accuracy, y-uniform 2D == 1D, flux == advection
  API                 — unknown scheme -> ValueError; factories pass scheme
"""

import numpy as np
import pytest

from upde import PDE, PDESystem, Burgers, AdvectionDiffusion
from upde.upde import (_limited_slope, _build_sparsity, _LIMITED_SCHEMES,
                       _convect_limited_1d, _flux_limited_1d)

LIMITERS = list(_LIMITED_SCHEMES)


def _square(x):
    return np.where((x > 0.25) & (x < 0.5), 1.0, 0.0)


def _smooth(x):
    return np.sin(np.pi * x) ** 4


def _advect_periodic(scheme, ic, n=200, c=1.0, T=1.0):
    """One full period of linear advection on a periodic unit domain."""
    x  = np.linspace(0.0, 1.0, n, endpoint=False)
    eq = PDE("u", x=x)
    eq.add_advection(velocity=c, scheme=scheme)
    eq.set_bc(kind="periodic")
    eq.set_ic(ic)
    sol = eq.solve(t_span=(0.0, T), method="RK45", rtol=1e-6, atol=1e-9)
    return sol.u[:, -1], ic(x)


# ---------------------------------------------------------------------------
# Limiter functions
# ---------------------------------------------------------------------------

class TestLimitedSlope:

    # psi(r) for r = dm/dp, with dp = 1
    @pytest.mark.parametrize("limiter, r, psi", [
        ("minmod",   0.5, 0.5),  ("minmod",   2.0, 1.0),
        ("van_leer", 0.5, 2/3),  ("van_leer", 2.0, 4/3),
        ("superbee", 0.5, 1.0),  ("superbee", 2.0, 2.0),
        ("mc",       0.5, 0.75), ("mc",       2.0, 1.5),
        ("mc",       0.2, 0.4),  ("mc",       5.0, 2.0),
    ])
    def test_psi_values(self, limiter, r, psi):
        s = _limited_slope(np.array([r]), np.array([1.0]), limiter)
        assert s[0] == pytest.approx(psi)

    @pytest.mark.parametrize("limiter", LIMITERS)
    def test_zero_at_extrema(self, limiter):
        dm = np.array([1.0, -1.0, 0.0, 2.0])
        dp = np.array([-1.0, 1.0, 3.0, 0.0])
        np.testing.assert_array_equal(_limited_slope(dm, dp, limiter), 0.0)

    @pytest.mark.parametrize("limiter", LIMITERS)
    def test_symmetric(self, limiter):
        dm = np.array([0.3, 1.0, 2.5])
        dp = np.array([1.0, 0.4, 1.0])
        np.testing.assert_allclose(_limited_slope(dm, dp, limiter),
                                   _limited_slope(dp, dm, limiter))
        np.testing.assert_allclose(_limited_slope(-dm, -dp, limiter),
                                   -_limited_slope(dm, dp, limiter))


# ---------------------------------------------------------------------------
# Linear advection
# ---------------------------------------------------------------------------

class TestLinearAdvection:

    @pytest.mark.parametrize("limiter", LIMITERS)
    def test_tvd_square_wave(self, limiter):
        u, _ = _advect_periodic(limiter, _square)
        assert u.min() >= -1e-8
        assert u.max() <= 1.0 + 1e-8

    @pytest.mark.parametrize("limiter", LIMITERS)
    def test_beats_upwind(self, limiter):
        u_up, u0 = _advect_periodic("upwind", _square)
        u_lim, _ = _advect_periodic(limiter, _square)
        assert np.mean(np.abs(u_lim - u0)) < 0.5 * np.mean(np.abs(u_up - u0))

    @pytest.mark.parametrize("limiter", ["van_leer", "mc"])
    def test_second_order_smooth(self, limiter):
        errs = []
        for n in (100, 200, 400):
            u, u0 = _advect_periodic(limiter, _smooth, n=n)
            errs.append(np.mean(np.abs(u - u0)))
        orders = np.log2(np.array(errs[:-1]) / np.array(errs[1:]))
        assert np.all(orders > 1.8), orders

    def test_central_second_order_smooth(self):
        errs = []
        for n in (100, 200):
            u, u0 = _advect_periodic("central", _smooth, n=n)
            errs.append(np.mean(np.abs(u - u0)))
        assert np.log2(errs[0] / errs[1]) > 1.9

    @pytest.mark.parametrize("limiter", LIMITERS)
    def test_negative_velocity_mirrors(self, limiter):
        u_pos, _ = _advect_periodic(limiter, _square, c=+1.0, T=0.25)
        u_neg, _ = _advect_periodic(limiter, lambda x: _square(1.0 - x),
                                    c=-1.0, T=0.25)
        # u_neg(x) should equal u_pos(1 - x), up to the grid shift
        np.testing.assert_allclose(np.roll(u_neg[::-1], 1), u_pos, atol=1e-6)

    def test_dirichlet_inflow_open_outflow(self):
        x  = np.linspace(0.0, 1.0, 201)
        ic = lambda x: np.exp(-200.0 * (x - 0.3) ** 2)
        eq = PDE("u", x=x)
        eq.add_advection(velocity=1.0, scheme="van_leer")
        eq.set_bc(side="left", kind="dirichlet", value=0.0)
        eq.set_ic(ic)
        sol = eq.solve(t_span=(0.0, 0.4), rtol=1e-6, atol=1e-9)
        assert np.mean(np.abs(sol.u[:, -1] - ic(x - 0.4))) < 3e-3

    @pytest.mark.parametrize("method", ["BDF", "Radau"])
    def test_implicit_methods(self, method):
        x  = np.linspace(0.0, 1.0, 100, endpoint=False)
        eq = PDE("u", x=x)
        eq.add_advection(velocity=1.0, scheme="van_leer")
        eq.set_bc(kind="periodic")
        eq.set_ic(_square)
        sol = eq.solve(t_span=(0.0, 0.5), method=method, rtol=1e-6, atol=1e-9)
        assert sol.success
        assert -1e-6 <= sol.u[:, -1].min() and sol.u[:, -1].max() <= 1 + 1e-6


# ---------------------------------------------------------------------------
# Flux form
# ---------------------------------------------------------------------------

class TestFluxForm:

    @pytest.mark.parametrize("c", [1.5, -0.7])
    @pytest.mark.parametrize("limiter", LIMITERS)
    def test_linear_flux_matches_advection(self, limiter, c):
        """For F = c*u the flux and convective operators must coincide."""
        rng = np.random.default_rng(0)
        phi = rng.standard_normal(50)
        for bc in ("periodic", "none"):
            conv = _convect_limited_1d(phi, c, 0.1, limiter, bc, bc)
            flux = _flux_limited_1d(lambda u: c * u, phi, 0.1, limiter, bc, bc)
            np.testing.assert_allclose(flux, conv, rtol=1e-6, atol=1e-8)

    @pytest.mark.parametrize("limiter", LIMITERS)
    def test_burgers_sonic_expansion(self, limiter):
        """uL=-1, uR=1: exact solution u = x/t in |x| < t (no shock at 0)."""
        x  = np.linspace(-1.0, 1.0, 401)
        T  = 0.5
        eq = Burgers("u", x=x, scheme=limiter)
        eq.set_bc(side="left",  kind="dirichlet", value=-1.0)
        eq.set_bc(side="right", kind="dirichlet", value=1.0)
        eq.set_ic(lambda x: np.where(x < 0.0, -1.0, 1.0))
        sol = eq.solve(t_span=(0.0, T), rtol=1e-6, atol=1e-9)
        exact = np.clip(x / T, -1.0, 1.0)
        assert np.mean(np.abs(sol.u[:, -1] - exact)) < 1e-2

    @pytest.mark.parametrize("limiter", LIMITERS)
    def test_burgers_shock_speed(self, limiter):
        """uL=1, uR=0: shock moves at speed 1/2, no overshoot."""
        x  = np.linspace(-1.0, 1.0, 401)
        T  = 0.5
        eq = Burgers("u", x=x, scheme=limiter)
        eq.set_bc(side="left",  kind="dirichlet", value=1.0)
        eq.set_bc(side="right", kind="dirichlet", value=0.0)
        eq.set_ic(lambda x: np.where(x < 0.0, 1.0, 0.0))
        sol = eq.solve(t_span=(0.0, T), rtol=1e-6, atol=1e-9)
        u = sol.u[:, -1]
        assert np.mean(np.abs(u - np.where(x < 0.5 * T, 1.0, 0.0))) < 2e-3
        assert u.min() >= -1e-6 and u.max() <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# Jacobian sparsity and steady solves
# ---------------------------------------------------------------------------

class TestSparsityAndSteady:

    def _pattern(self, scheme, periodic):
        x  = np.linspace(0.0, 1.0, 10)
        eq = PDE("u", x=x)
        eq.add_advection(velocity=1.0, scheme=scheme)
        if periodic:
            eq.set_bc(kind="periodic")
        return _build_sparsity([eq]).toarray()

    def test_upwind_tridiagonal(self):
        S = self._pattern("upwind", periodic=False)
        assert S[5, 3] == 0 and S[5, 4] == 1 and S[5, 6] == 1

    def test_limited_pentadiagonal(self):
        S = self._pattern("van_leer", periodic=False)
        assert S[5, 3] == 1 and S[5, 7] == 1 and S[5, 8] == 0
        assert S[0, 2] == 1 and S[0, 9] == 0

    def test_periodic_wraps(self):
        S = self._pattern("upwind", periodic=True)
        assert S[0, 9] == 1 and S[9, 0] == 1
        S = self._pattern("van_leer", periodic=True)
        assert S[0, 8] == 1 and S[1, 9] == 1 and S[9, 1] == 1

    @staticmethod
    def _numerical_jacobian(eq, rng):
        sysm = PDESystem([eq])
        y0   = rng.standard_normal(eq.n_total)
        f0   = sysm._rhs(0.0, y0)
        J    = np.zeros((len(f0), len(y0)))
        for k in range(len(y0)):
            yp = y0.copy(); yp[k] += 1e-6
            J[:, k] = (sysm._rhs(0.0, yp) - f0) / 1e-6
        return J

    @pytest.mark.parametrize("periodic", [False, True])
    @pytest.mark.parametrize("scheme", ["upwind", "central"] + LIMITERS)
    @pytest.mark.parametrize("dim", [1, 2])
    def test_pattern_covers_jacobian(self, dim, scheme, periodic):
        x  = np.linspace(0.0, 1.0, 9)
        if dim == 1:
            eq = PDE("u", x=x)
            eq.add_advection(velocity=lambda x: np.sin(5 * x), scheme=scheme)
            eq.add_flux(flux=lambda u: 0.5 * u**2, scheme=scheme)
        else:
            eq = PDE("u", x=x, y=x[:7])
            eq.add_advection(velocity_x=1.0, velocity_y=-0.5, scheme=scheme)
            eq.add_flux(flux_x=lambda u: 0.5 * u**2,
                        flux_y=lambda u: u**3 / 3, scheme=scheme)
        eq.add_diffusion(diffusivity=0.1)
        if periodic:
            eq.set_bc(kind="periodic", side="all") if dim == 2 \
                else eq.set_bc(kind="periodic")
        J = self._numerical_jacobian(eq, np.random.default_rng(1))
        S = _build_sparsity([eq]).toarray()
        missing = (np.abs(J) > 1e-8 * np.abs(J).max()) & (S == 0)
        assert not missing.any(), f"{missing.sum()} Jacobian entries outside pattern"

    def test_steady_auto_detects_nonlinear(self):
        x  = np.linspace(0.0, 1.0, 101)
        Pe = 50.0
        eq = PDE("u", x=x)
        eq.add_advection(velocity=1.0, scheme="van_leer")
        eq.add_diffusion(diffusivity=1.0 / Pe)
        eq.set_bc(side="left",  kind="dirichlet", value=0.0)
        eq.set_bc(side="right", kind="dirichlet", value=1.0)
        sol = eq.solve_steady()
        assert sol.solver == "nonlinear"
        assert sol.success
        exact = (np.exp(Pe * (x - 1.0)) - np.exp(-Pe)) / (1.0 - np.exp(-Pe))
        assert np.max(np.abs(sol.u - exact)) < 2e-2


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

class TestSchemeAPI:

    def test_unknown_scheme(self):
        eq = PDE("u", x=np.linspace(0, 1, 10))
        with pytest.raises(ValueError, match="scheme"):
            eq.add_advection(velocity=1.0, scheme="lax_wendroff")
        with pytest.raises(ValueError, match="scheme"):
            eq.add_flux(flux=lambda u: u, scheme="lax_wendroff")


    def test_default_scheme_is_upwind(self):
        eq = PDE("u", x=np.linspace(0, 1, 10))
        eq.add_advection(velocity=1.0)
        eq.add_flux(flux=lambda u: u)
        assert eq._advection[0]["scheme"] == "upwind"
        assert eq._fluxes[0]["scheme"] == "upwind"

    def test_factories_pass_scheme(self):
        x = np.linspace(0, 1, 10)
        assert Burgers("u", x, scheme="mc")._fluxes[0]["scheme"] == "mc"
        eq = AdvectionDiffusion("T", x, velocity=1.0, diffusivity=0.01,
                                scheme="minmod")
        assert eq._advection[0]["scheme"] == "minmod"


# ---------------------------------------------------------------------------
# 2D
# ---------------------------------------------------------------------------

def _advect_periodic_2d(scheme, ic, n=48, T=1.0):
    """Diagonal advection (cx = cy = 1) for one period on the unit square."""
    x  = np.linspace(0.0, 1.0, n, endpoint=False)
    eq = PDE("u", x=x, y=x)
    eq.add_advection(velocity_x=1.0, velocity_y=1.0, scheme=scheme)
    eq.set_bc(kind="periodic", side="all")
    eq.set_ic(ic)
    sol = eq.solve(t_span=(0.0, T), rtol=1e-6, atol=1e-9)
    X, Y = np.meshgrid(x, x, indexing="ij")
    return sol.u[:, :, -1], ic(X, Y)


def _square_2d(X, Y):
    return np.where((np.abs(X - 0.35) < 0.15) & (np.abs(Y - 0.35) < 0.15), 1.0, 0.0)


class Test2D:

    @pytest.mark.parametrize("limiter", LIMITERS)
    def test_tvd_and_beats_upwind(self, limiter):
        u_up, u0 = _advect_periodic_2d("upwind", _square_2d)
        u, _     = _advect_periodic_2d(limiter, _square_2d)
        assert u.min() >= -1e-8 and u.max() <= 1.0 + 1e-8
        assert np.mean(np.abs(u - u0)) < 0.6 * np.mean(np.abs(u_up - u0))

    @pytest.mark.parametrize("limiter", LIMITERS)
    @pytest.mark.parametrize("form", ["advection", "flux"])
    def test_y_uniform_matches_1d(self, limiter, form):
        """A field constant in y must give the 1D operator on every row."""
        x, ny = np.linspace(0.0, 1.0, 40), 5
        U  = np.where(np.abs(x - 0.4) < 0.15, 1.0, 0.2) + 0.1 * np.sin(7 * x)
        e1 = PDE("u", x=x)
        e2 = PDE("u", x=x, y=np.linspace(0.0, 1.0, ny))
        if form == "advection":
            e1.add_advection(velocity=0.7, scheme=limiter)
            e2.add_advection(velocity_x=0.7, velocity_y=0.3, scheme=limiter)
        else:
            e1.add_flux(flux=lambda u: 0.5 * u**2, scheme=limiter)
            e2.add_flux(flux_x=lambda u: 0.5 * u**2, flux_y=lambda u: u,
                        scheme=limiter)
        r1 = PDESystem([e1])._rhs(0.0, U)
        r2 = PDESystem([e2])._rhs(0.0, np.repeat(U[:, None], ny, 1).ravel())
        np.testing.assert_allclose(r2.reshape(len(x), ny), r1[:, None] * np.ones(ny),
                                   atol=1e-10)

    @pytest.mark.parametrize("limiter", LIMITERS)
    def test_linear_flux_matches_advection_2d(self, limiter):
        x   = np.linspace(0.0, 1.0, 12)
        y   = np.linspace(0.0, 1.0, 9)
        phi = np.random.default_rng(3).standard_normal((12, 9)).ravel()
        rs  = []
        for flux in (False, True):
            eq = PDE("u", x=x, y=y)
            if flux:
                eq.add_flux(flux_x=lambda u: 1.3 * u, flux_y=lambda u: -0.6 * u,
                            scheme=limiter)
            else:
                eq.add_advection(velocity_x=1.3, velocity_y=-0.6, scheme=limiter)
            eq.set_bc(kind="periodic", side="x")
            rs.append(PDESystem([eq])._rhs(0.0, phi))
        np.testing.assert_allclose(rs[1], rs[0], rtol=1e-6, atol=1e-8)
