"""
conftest.py — shared fixtures and snapshot (golden-file) infrastructure.

Golden-file workflow
--------------------
  First run (no snapshots exist):
      pytest                    # snapshots are created, tests pass silently

  Normal run (snapshots exist):
      pytest                    # diffs against snapshots; any mismatch = FAIL

  After an intentional change:
      pytest --rebless          # overwrites snapshots with new results
      git diff tests/snapshots/ # review what changed before committing

Tolerances:
      rtol=1e-6, atol=1e-10    # tight enough to catch real regressions
"""

import numpy as np
import pathlib
import pytest
SNAPSHOT_DIR = pathlib.Path(__file__).parent / "snapshots"


# ---------------------------------------------------------------------------
# CLI flag
# ---------------------------------------------------------------------------

def pytest_addoption(parser):
    parser.addoption(
        "--rebless",
        action="store_true",
        default=False,
        help="Overwrite golden snapshot files with current results.",
    )


@pytest.fixture
def rebless(request):
    return request.config.getoption("--rebless")


# ---------------------------------------------------------------------------
# Snapshot helper
# ---------------------------------------------------------------------------

class Snapshot:
    """
    Compare or create a .npz golden file for a dict of named numpy arrays.

    Usage
    -----
        def test_foo(snapshot):
            sol = run_my_pde(...)
            snapshot.check("test_foo", {"u_final": sol.u[:, -1]})
    """

    RTOL = 1e-6
    ATOL = 1e-10

    def __init__(self, rebless: bool):
        self._rebless = rebless
        SNAPSHOT_DIR.mkdir(exist_ok=True)

    def check(self, name: str, arrays: dict):
        path = SNAPSHOT_DIR / f"{name}.npz"

        # --rebless: overwrite and skip (not fail) so the run stays green.
        if self._rebless:
            np.savez_compressed(path, **arrays)
            pytest.skip(f"Reblessed → {path.name}")

        # First ever run: create the snapshot, test passes silently.
        if not path.exists():
            np.savez_compressed(path, **arrays)
            return

        gold = np.load(path)

        missing = set(arrays) - set(gold.files)
        extra   = set(gold.files) - set(arrays)
        assert not missing, (
            f"Snapshot '{name}' is missing keys: {missing}. "
            "Run --rebless if this is intentional."
        )
        assert not extra, (
            f"Snapshot '{name}' has unexpected keys: {extra}. "
            "Run --rebless if this is intentional."
        )

        for key, arr in arrays.items():
            np.testing.assert_allclose(
                arr, gold[key],
                rtol=self.RTOL,
                atol=self.ATOL,
                err_msg=f"Snapshot mismatch in '{name}' → key='{key}'",
            )


@pytest.fixture
def snapshot(rebless):
    return Snapshot(rebless)


# ---------------------------------------------------------------------------
# Shared grids  (reused across test files for speed and consistency)
# ---------------------------------------------------------------------------

@pytest.fixture
def grid_1d_32():
    return np.linspace(0, 1, 32)


@pytest.fixture
def grid_1d_64():
    return np.linspace(0, 1, 64)


@pytest.fixture
def grid_2d_16():
    x = np.linspace(0, 1, 16)
    y = np.linspace(0, 1, 16)
    return x, y
