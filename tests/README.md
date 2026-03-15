# Testing uPDE

## Setup

```bash
pip install pytest pytest-cov
```

Run from the repo root (`uPDE_git/`):

```bash
pytest
```

The first run creates `tests/snapshots/*.npz` — the golden files. Commit them:

```bash
git add tests/snapshots/
git commit -m "add golden snapshots"
```

---

## Daily workflow

Run the full suite after any change to `upde.py` or `equations.py`:

```bash
pytest
```

| Result | Meaning | Action |
|--------|---------|--------|
| All green | Nothing changed numerically | Commit your code |
| Red | Numbers drifted | See below |

---

## When tests go red

**It's a bug** — fix the code, run `pytest` again.

**It's intentional** (you improved a stencil, fixed a BC, added a feature that changes output) — rebless the snapshots:

```bash
pytest --rebless
git diff tests/snapshots/   # review what changed
git add tests/snapshots/
git commit -m "rebless snapshots: <reason>"
```

Always write the reason in the commit message so future-you knows why the numbers changed.

---

## Adding a new test

### Regression test (most common)
Add a function to `tests/test_regression.py`:

```python
def test_reg_my_new_case(snapshot):
    eq = PDE('u', x=np.linspace(0, 1, 32))
    # ... set up your problem ...
    sol = eq.solve((0, 1.0), method='RK45')
    assert sol.success
    snapshot.check("reg_my_new_case", {"u_final": sol.u[:, -1]})
```

Run `pytest` — the snapshot is created automatically on the first run. Commit it.

### Physics test (when you have an analytical answer)
Add a function to `tests/test_physics.py`:

```python
def test_my_analytical_case():
    # solve, compare against exact solution
    assert max_err < 1e-3
```

### Unit test (API behaviour, guards, edge cases)
Add a function to `tests/test_core.py`.

---

## Useful commands

```bash
pytest                          # run everything
pytest -k reg                   # only regression tests
pytest -k "not reg"             # skip regression tests (faster)
pytest --rebless                # regenerate all snapshots
pytest --cov=upde --cov-report=term-missing   # coverage report
```
