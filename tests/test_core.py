"""
test_core.py — unit tests for upde.py internals and the validation layer.

Covers
------
  _call()       — scalar, t injection (explicit param + VAR_KEYWORD), field
                  filtering, default t=0, no-t backward compat
  PDE()         — type guard, 't' name reservation, 1D/2D construction
  PDESystem()   — duplicate fields, grid mismatch, undeclared field reference
  set_ic()      — scalar, array, callable
"""

import numpy as np
import pytest

from upde import PDE, PDESystem
from upde.upde import _call, _to_callable


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _x(n=32):
    return np.linspace(0, 1, n)

def _coords(n=32):
    return (_x(n),)

def _fields():
    return {"u": np.ones(32), "v": np.zeros(32)}

def _simple_eq(name="u", n=16):
    """Minimal valid PDE for validation tests."""
    x = _x(n)
    eq = PDE(name, x=x)
    eq.add_diffusion(diffusivity=0.01)
    eq.set_bc(side="left",  kind="dirichlet", value=0.0)
    eq.set_bc(side="right", kind="dirichlet", value=0.0)
    eq.set_ic(0.0)
    return eq


# ---------------------------------------------------------------------------
# _call — parameter injection
# ---------------------------------------------------------------------------

class TestCallInjection:

    def test_scalar_callable_broadcasts(self):
        fn     = _to_callable(3.14)
        result = _call(fn, _coords(), _fields(), t=0.0)
        # _to_callable broadcasts a scalar to a constant array
        assert np.all(result == pytest.approx(3.14))

    def test_no_t_param_not_injected(self):
        """Callable without 't' must not receive it — no TypeError."""
        def fn(x, u):
            return u
        # Would raise TypeError if t were forwarded as an unexpected kwarg.
        _call(fn, _coords(), _fields(), t=99.0)

    def test_explicit_t_param_injected(self):
        received = {}
        def fn(x, t):
            received["t"] = t
            return np.zeros_like(x)
        _call(fn, _coords(), _fields(), t=42.0)
        assert received["t"] == pytest.approx(42.0)

    def test_var_keyword_receives_t(self):
        received = {}
        def fn(x, **kw):
            received.update(kw)
            return np.zeros_like(x)
        _call(fn, _coords(), _fields(), t=7.5)
        assert received["t"] == pytest.approx(7.5)

    def test_var_keyword_receives_all_fields(self):
        received = {}
        def fn(x, **kw):
            received.update(kw)
            return np.zeros_like(x)
        _call(fn, _coords(), _fields(), t=0.0)
        assert "u" in received
        assert "v" in received

    def test_selective_field_injection(self):
        """Only the parameters the callable declares are forwarded."""
        injected = {}
        def fn(x, u):
            injected["keys"] = list(locals().keys())
            return u
        _call(fn, _coords(), _fields(), t=0.0)
        assert "u" in injected["keys"]
        assert "v" not in injected["keys"]

    def test_t_defaults_to_zero(self):
        """Calling _call without t= should default to t=0.0."""
        received = {}
        def fn(x, t):
            received["t"] = t
            return np.zeros_like(x)
        _call(fn, _coords(), _fields())    # intentionally no t=
        assert received["t"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# PDE construction guards
# ---------------------------------------------------------------------------

class TestPDEConstruction:

    def test_field_must_be_string(self):
        with pytest.raises(TypeError):
            PDE(42, x=_x())

    def test_field_t_reserved(self):
        with pytest.raises(ValueError, match="reserved"):
            PDE("t", x=_x())

    def test_valid_1d(self):
        eq = PDE("T", x=_x())
        assert eq.field == "T"
        assert not eq.is_2d
        assert eq.nx == 32

    def test_valid_2d(self):
        x = _x(8)
        eq = PDE("T", x=x, y=x)
        assert eq.is_2d
        assert eq.X.shape == (8, 8)
        assert eq.Y.shape == (8, 8)


# ---------------------------------------------------------------------------
# PDESystem validation
# ---------------------------------------------------------------------------

class TestPDESystemValidation:

    def test_duplicate_field_names_raise(self):
        with pytest.raises(ValueError, match="[Dd]uplicate|duplicate"):
            PDESystem([_simple_eq("u"), _simple_eq("u")])

    def test_grid_mismatch_raises(self):
        with pytest.raises((ValueError, AssertionError)):
            PDESystem([_simple_eq("u", n=16), _simple_eq("v", n=32)])

    def test_undeclared_field_reference_raises(self):
        x  = _x()
        eq = PDE("u", x=x)
        eq.add_source(expr=lambda x, ghost: ghost)   # 'ghost' not declared
        eq.set_bc(side="left",  kind="dirichlet", value=0.0)
        eq.set_bc(side="right", kind="dirichlet", value=0.0)
        eq.set_ic(0.0)
        with pytest.raises(ValueError):
            PDESystem([eq])


# ---------------------------------------------------------------------------
# set_ic edge cases
# ---------------------------------------------------------------------------

class TestInitialConditions:

    def _eq(self, x=None):
        if x is None:
            x = _x(16)
        eq = PDE("u", x=x)
        eq.add_diffusion(diffusivity=0.01)
        eq.set_bc(side="left",  kind="dirichlet", value=0.0)
        eq.set_bc(side="right", kind="dirichlet", value=0.0)
        return eq

    def test_scalar_ic(self):
        eq = self._eq()
        eq.set_ic(2.0)
        sol = eq.solve((0, 0.01), method="RK45")
        assert sol.success

    def test_array_ic(self):
        x  = _x(16)
        eq = self._eq(x)
        eq.set_ic(np.sin(np.pi * x))
        sol = eq.solve((0, 0.01), method="RK45")
        assert sol.success

    def test_callable_ic(self):
        x  = _x(16)
        eq = self._eq(x)
        eq.set_ic(lambda x: np.sin(np.pi * x))
        sol = eq.solve((0, 0.01), method="RK45")
        assert sol.success
