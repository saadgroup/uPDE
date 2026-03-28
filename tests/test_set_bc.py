"""
test_set_bc.py — tests for the extended set_bc API.

Covers
------
  side='all'          — broadcasts scalar value to all walls (2D and 1D)
  side=list           — sets multiple sides in one call, same or different values
  override semantics  — named-side registration removes prior BC for that wall
  mask= layering      — mask= always appends, enabling intentional overlap
  error cases         — side='all' + list, mismatched list lengths
  physics             — end-to-end solve confirms override produces correct result
"""

import numpy as np
import pytest

from upde import PDE, HeatEquation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _eq2d(n=8):
    x = np.linspace(0, 1, n)
    return PDE("T", x=x, y=x)

def _eq1d(n=16):
    return PDE("T", x=np.linspace(0, 1, n))


# ---------------------------------------------------------------------------
# side='all'
# ---------------------------------------------------------------------------

class TestSideAll:

    def test_2d_registers_four_sides(self):
        eq = _eq2d()
        eq.set_bc(side="all", kind="dirichlet", value=0.0)
        assert len(eq._bcs) == 4
        assert {bc.normal for bc in eq._bcs} == {"x-", "x+", "y-", "y+"}

    def test_2d_all_have_correct_value(self):
        eq = _eq2d()
        eq.set_bc(side="all", kind="dirichlet", value=3.14)
        assert all(bc.value == pytest.approx(3.14) for bc in eq._bcs)

    def test_1d_registers_two_sides(self):
        eq = _eq1d()
        eq.set_bc(side="all", kind="dirichlet", value=0.0)
        assert len(eq._bcs) == 2
        assert {bc.normal for bc in eq._bcs} == {"x-", "x+"}

    def test_all_with_list_value_raises(self):
        eq = _eq2d()
        with pytest.raises(ValueError, match="ambiguous"):
            eq.set_bc(side="all", kind="dirichlet", value=[1.0, 0.0, 0.0, 0.0])

    def test_all_callable_value(self):
        eq = _eq2d()
        fn = lambda t: np.sin(t)
        eq.set_bc(side="all", kind="dirichlet", value=fn)
        assert all(bc.value is fn for bc in eq._bcs)


# ---------------------------------------------------------------------------
# side=list
# ---------------------------------------------------------------------------

class TestSideList:

    def test_list_same_scalar_value(self):
        eq = _eq2d()
        eq.set_bc(side=["left", "right"], kind="dirichlet", value=1.0)
        assert len(eq._bcs) == 2
        assert {bc.normal for bc in eq._bcs} == {"x-", "x+"}
        assert all(bc.value == pytest.approx(1.0) for bc in eq._bcs)

    def test_list_different_values(self):
        eq = _eq2d()
        eq.set_bc(side=["left", "right"], kind="dirichlet", value=[1.0, 0.0])
        vals = {bc.normal: bc.value for bc in eq._bcs}
        assert vals["x-"] == pytest.approx(1.0)
        assert vals["x+"] == pytest.approx(0.0)

    def test_list_mismatched_lengths_raises(self):
        eq = _eq2d()
        with pytest.raises(ValueError, match="must match"):
            eq.set_bc(side=["left", "right"], kind="dirichlet", value=[1.0, 0.0, 0.5])

    def test_list_neumann(self):
        eq = _eq2d()
        eq.set_bc(side=["bottom", "top"], kind="neumann", value=0.0)
        assert len(eq._bcs) == 2
        assert {bc.normal for bc in eq._bcs} == {"y-", "y+"}
        assert all(bc.kind == "neumann" for bc in eq._bcs)


# ---------------------------------------------------------------------------
# Override semantics
# ---------------------------------------------------------------------------

class TestOverride:

    def test_second_call_same_side_replaces_first(self):
        eq = _eq2d()
        eq.set_bc(side="left", kind="dirichlet", value=1.0)
        eq.set_bc(side="left", kind="dirichlet", value=5.0)
        left = [bc for bc in eq._bcs if bc.normal == "x-"]
        assert len(left) == 1
        assert left[0].value == pytest.approx(5.0)

    def test_override_changes_kind(self):
        eq = _eq2d()
        eq.set_bc(side="top", kind="dirichlet", value=1.0)
        eq.set_bc(side="top", kind="neumann",   value=0.0)
        top = [bc for bc in eq._bcs if bc.normal == "y+"]
        assert len(top) == 1
        assert top[0].kind == "neumann"

    def test_all_then_override_one(self):
        """set_bc(all) then override one side → exactly 4 BCs, correct kinds."""
        eq = _eq2d()
        eq.set_bc(side="all", kind="dirichlet", value=0.0)
        eq.set_bc(side="top", kind="neumann",   value=0.0)
        assert len(eq._bcs) == 4
        kinds = {bc.normal: bc.kind for bc in eq._bcs}
        assert kinds["x-"] == "dirichlet"
        assert kinds["x+"] == "dirichlet"
        assert kinds["y-"] == "dirichlet"
        assert kinds["y+"] == "neumann"

    def test_all_then_override_two(self):
        """Override two sides independently after side='all'."""
        eq = _eq2d()
        eq.set_bc(side="all",    kind="dirichlet", value=0.0)
        eq.set_bc(side="top",    kind="neumann",   value=0.0)
        eq.set_bc(side="bottom", kind="neumann",   value=0.0)
        assert len(eq._bcs) == 4
        kinds = {bc.normal: bc.kind for bc in eq._bcs}
        assert kinds["x-"] == "dirichlet"
        assert kinds["x+"] == "dirichlet"
        assert kinds["y-"] == "neumann"
        assert kinds["y+"] == "neumann"

    def test_list_then_override_one(self):
        """set_bc with a list then override one of those sides."""
        eq = _eq2d()
        eq.set_bc(side=["left", "right"], kind="dirichlet", value=1.0)
        eq.set_bc(side="right", kind="dirichlet", value=0.0)
        vals = {bc.normal: bc.value for bc in eq._bcs}
        assert vals["x-"] == pytest.approx(1.0)
        assert vals["x+"] == pytest.approx(0.0)
        assert len(eq._bcs) == 2

    def test_no_stale_bc_after_override(self):
        """Overriding must not leave ghost entries in _bcs."""
        eq = _eq2d()
        eq.set_bc(side="all",  kind="dirichlet", value=1.0)
        eq.set_bc(side="top",  kind="neumann",   value=0.0)
        # old dirichlet on top must be gone
        assert ("dirichlet", "y+") not in [(bc.kind, bc.normal) for bc in eq._bcs]


# ---------------------------------------------------------------------------
# mask= layering (no deduplication)
# ---------------------------------------------------------------------------

class TestMaskLayering:

    def test_mask_appends_alongside_named_side(self):
        """mask= does not remove the full-wall BC — both coexist."""
        eq = _eq2d(n=10)
        inlet = np.zeros((10, 10), dtype=bool)
        inlet[0, 3:7] = True
        eq.set_bc(side="left",  kind="dirichlet", value=0.0)
        eq.set_bc(mask=inlet,   kind="dirichlet", value=1.0)
        assert len(eq._bcs) == 2
        normals = [bc.normal for bc in eq._bcs]
        assert "x-" in normals   # full-wall entry
        assert None in normals   # mask entry


# ---------------------------------------------------------------------------
# Physics: end-to-end correctness of override pattern
# ---------------------------------------------------------------------------

class TestOverridePhysics:

    def test_all_then_override_right_gives_linear_steady_state(self, grid_1d_64):
        """
        Set all sides to Dirichlet=1 via side='all', then override right=0.
        Steady state of the 1D heat equation must be u = 1 - x,
        identical to setting left=1 and right=0 individually.
        """
        x  = grid_1d_64
        eq = PDE("T", x=x)
        eq.add_diffusion(diffusivity=0.1)
        eq.set_bc(side="all",   kind="dirichlet", value=1.0)
        eq.set_bc(side="right", kind="dirichlet", value=0.0)
        eq.set_ic(0.0)
        sol = eq.solve((0, 10.0), method="RK45", rtol=1e-8, atol=1e-10)
        assert sol.success
        err = np.max(np.abs(sol.T[:, -1] - (1.0 - x)))
        assert err < 1e-3, f"Expected linear profile; L-inf error = {err:.2e}"

    def test_list_sides_gives_same_result_as_individual_calls(self, grid_1d_64):
        """
        set_bc(side=['left','right'], value=[1.0, 0.0]) must produce the
        same steady state as two individual set_bc calls.
        """
        x = grid_1d_64

        eq_list = PDE("T", x=x)
        eq_list.add_diffusion(diffusivity=0.1)
        eq_list.set_bc(side=["left", "right"], kind="dirichlet", value=[1.0, 0.0])
        eq_list.set_ic(0.0)
        sol_list = eq_list.solve((0, 10.0), method="RK45", rtol=1e-8, atol=1e-10)

        eq_ind = PDE("T", x=x)
        eq_ind.add_diffusion(diffusivity=0.1)
        eq_ind.set_bc(side="left",  kind="dirichlet", value=1.0)
        eq_ind.set_bc(side="right", kind="dirichlet", value=0.0)
        eq_ind.set_ic(0.0)
        sol_ind = eq_ind.solve((0, 10.0), method="RK45", rtol=1e-8, atol=1e-10)

        assert sol_list.success and sol_ind.success
        diff = np.max(np.abs(sol_list.T[:, -1] - sol_ind.T[:, -1]))
        assert diff < 1e-10, f"list vs individual set_bc mismatch: {diff:.2e}"
