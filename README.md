<!-- Last modified: 2026-03-24 14:51 UTC -->
# uPDE

> A lightweight Python library for solving 1D and 2D PDEs using the method of lines.

uPDE lets you describe a PDE — its terms, coefficients, boundary conditions, and initial
data — and delegates the hard work to SciPy. Transient problems go to
`scipy.integrate.solve_ivp`; steady-state problems go to `scipy.optimize.root`.
You get adaptive time-stepping, stiffness detection, and Newton iteration without
writing any solver code yourself.

**Features:**
- 1D and 2D problems on uniform Cartesian grids
- Coupled multi-field systems
- Scalar, array, and callable (nonlinear, field-dependent, or time-dependent) coefficients
- Dirichlet, Neumann, and periodic boundary conditions
- Interior boundary conditions for obstacles and inclusions
- **Transient solver** — method of lines via `scipy.integrate.solve_ivp`
- **Steady-state solver** — `solve_steady()` via `scipy.optimize.root`; works for linear and nonlinear problems with no user-facing distinction
- Pre-built equation prototypes for common PDE families
- Flamelet-based combustion via mixture-fraction transport and `FlameletTable`
- Pure NumPy / SciPy — no compilation, no external mesh libraries

---

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
  - [Transient problem](#transient-problem)
  - [Steady-state problem](#steady-state-problem)
- [Core Concepts](#core-concepts)
- [User Guide](#user-guide)
  - [Coefficients and Callables](#coefficients-and-callables)
  - [Time-Dependent Callables](#time-dependent-callables)
  - [Boundary Conditions](#boundary-conditions)
  - [Initial Conditions](#initial-conditions)
  - [Coupled Systems](#coupled-systems)
  - [2D Problems](#2d-problems)
  - [Interior Boundary Conditions](#interior-boundary-conditions)
  - [Choosing a Transient Solver](#choosing-a-transient-solver)
  - [Steady-State Problems](#steady-state-problems)
- [Pre-built Equations](#pre-built-equations)
- [Combustion and Flamelet Chemistry](#combustion-and-flamelet-chemistry)
- [API Reference](#api-reference)
  - [PDE](#pde)
  - [PDESystem](#pdesystem)
  - [PDESolution](#pdesolution)
  - [SteadySolution](#steadysolution)
  - [Operator Reference](#operator-reference)
- [Numerical Methods](#numerical-methods)
- [Troubleshooting](#troubleshooting)

---

## Installation

uPDE has no compiled dependencies. Install from the repository:

```bash
git clone https://github.com/tsaad-dev/upde.git
cd upde
pip install -e .
```

Or drop the source files directly into your project:

```bash
cp upde.py equations.py chemistry.py your_project/
```

**Requirements:**

| Package | Version |
|---------|---------|
| numpy   | ≥ 2.0   |
| scipy   | ≥ 1.8   |
| python  | ≥ 3.9   |

**Optional — for high-fidelity flamelet table generation:**

| Package | Version | Purpose |
|---------|---------|---------|
| cantera | ≥ 3.0   | `FlameletTable.from_cantera()` — 1-D counterflow flame |

Cantera is only needed to *generate* a flamelet table. Once saved with
`table.save('ch4_air.npz')`, the table can be reloaded with
`FlameletTable.from_file()` on any machine without Cantera installed.

---

## Quick Start

### Transient problem

```python
import numpy as np
from upde import PDE, PDESystem

x = np.linspace(0, 1, 256)

eq = PDE('T', x=x)
eq.add_diffusion(diffusivity=0.05)
eq.set_bc(side='left',  kind='dirichlet', value=1.0)
eq.set_bc(side='right', kind='dirichlet', value=0.0)
eq.set_ic(0.0)

sol = eq.solve(t_span=(0, 1), method='BDF')
# sol.T         shape (256, nt)
# sol.T[:, -1]  final profile
# sol.t         time points
```

### Steady-state problem

```python
import numpy as np
from upde import PDE

x = np.linspace(0, 1, 256)

eq = PDE('T', x=x)
eq.add_diffusion(diffusivity=0.05)
eq.set_bc(side='left',  kind='dirichlet', value=1.0)
eq.set_bc(side='right', kind='dirichlet', value=0.0)

sol = eq.solve_steady()
# sol.T     shape (256,)  — no time dimension
# sol.residual   max|RHS| at solution
```

No initial condition is needed for steady-state problems. A nonlinear problem
is solved identically — just pass a reasonable starting guess:

```python
eq.add_diffusion(diffusivity=lambda x, T: 1 + T**2)   # nonlinear k
sol = eq.solve_steady(guess=np.linspace(1.0, 0.0, 256))
```

---

## Core Concepts

### PDE — equation descriptor

A `PDE` object holds the mathematical description of one equation: its field name,
spatial grid, terms (advection, diffusion, source, flux), boundary conditions, and
initial condition. It never solves anything on its own.

`PDE.solve()` and `PDE.solve_steady()` are available as convenience wrappers for
equations that do not reference any external field. For coupled equations, use
`PDESystem` explicitly.

### PDESystem — the solver

`PDESystem` takes a list of `PDE` objects, validates that all field references are
consistent, assembles the joint state vector, and delegates to SciPy:

- `PDESystem.solve(t_span, ...)` → calls `scipy.integrate.solve_ivp`
- `PDESystem.solve_steady(guess=None, ...)` → calls `scipy.optimize.root`

### PDESolution — transient result

Fields are accessed by name as attributes. Spatial indices come first, time last:

```python
sol.T           # (nx, nt) in 1D  |  (nx, ny, nt) in 2D
sol.T[:, -1]    # final 1D profile
sol.T[:, :, -1] # final 2D field
sol.t           # time points (nt,)
sol.success     # bool
sol.message     # integrator status string
sol.raw         # raw scipy OdeResult
```

### SteadySolution — steady-state result

```python
sol.T           # (nx,) in 1D  |  (nx, ny) in 2D  — no time dimension
sol.success     # bool (see note below)
sol.residual    # max|RHS(φ)| at solution — primary convergence indicator
sol.nfev        # number of RHS evaluations
sol.message     # convergence message
sol.raw         # raw scipy OptimizeResult
```

> **Note on `sol.success`:** `scipy.optimize.root` with `method='hybr'` (the default)
> occasionally reports `success=False` even when the solution is correct, particularly
> when starting from an all-zero guess. Always treat `sol.residual < 1e-8` as the
> authoritative convergence check.

---

## User Guide

### Coefficients and Callables

Every coefficient in uPDE — diffusivity, velocity, source — accepts four forms:

| Form | Example |
|------|---------|
| scalar | `0.05` |
| ndarray | `D_arr` — used as-is; must match the field shape |
| callable | `lambda x, T: 1 + T**2` — called at every RHS evaluation |
| string | `'u'` — resolved from the live coupled state (field coupling) |

**Callable signature convention**

Callables always receive coordinates first, then keyword arguments. Declare only
what you need — uPDE uses `inspect.signature` to inject only the parameters your
function actually declares:

| Dimension | Signature | Notes |
|-----------|-----------|-------|
| 1D | `f(x)` | coordinates only |
| 1D | `f(x, t)` | coordinates + current time |
| 1D | `f(x, fieldA)` | coordinates + one field |
| 1D | `f(x, t, fieldA)` | coordinates + time + one field |
| 1D | `f(x, **fields)` | coordinates + all fields |
| 2D | `f(x, y)` | coordinates only |
| 2D | `f(x, y, t)` | coordinates + time |
| 2D | `f(x, y, fieldA)` | coordinates + one field |
| 2D | `f(x, y, t, **fields)` | coordinates + time + all fields |

The ordering rule is: **coordinates → t → field names**. `t` is reserved and
cannot be used as a field name. Fixed parameters go in via closures:

```python
k1 = 10.0
eq.add_source(expr=lambda x, cA, cB: -k1 * cA * cB)  # k1 captured by closure
```

---

### Time-Dependent Callables

Declare `t` in your callable signature and uPDE injects the current solver time:

```python
# Source that shuts off at t = 0.1
eq.add_source(expr=lambda x, t: np.where(t < 0.1, 1.0, 0.0) * np.ones_like(x))

# Time-varying Dirichlet BC
eq.set_bc(side='left', kind='dirichlet', value=lambda t: np.sin(2 * np.pi * t))
```

Time-dependent callables work with transient solves. For `solve_steady()`, time is
fixed at `t=0` and is generally not meaningful.

---

### Boundary Conditions

#### Dirichlet — prescribed value

```python
eq.set_bc(side='left',  kind='dirichlet', value=1.0)
eq.set_bc(side='right', kind='dirichlet', value=0.0)

# time-varying
eq.set_bc(side='left', kind='dirichlet', value=lambda t: np.sin(t))
```

#### Neumann — prescribed normal derivative

```python
eq.set_bc(side='right', kind='neumann', value=0.0)   # zero-flux
eq.set_bc(side='left',  kind='neumann', value=2.5)   # prescribed flux
```

#### Periodic

```python
eq.set_bc(kind='periodic')                # 1D
eq.set_bc(kind='periodic', side='x')     # 2D — wrap left ↔ right
eq.set_bc(kind='periodic', side='y')     # 2D — wrap bottom ↔ top
eq.set_bc(kind='periodic', side='all')   # 2D — fully periodic
```

#### Side shortcuts

| Dimension | `side=` | Edge |
|-----------|---------|------|
| 1D | `'left'` | `x[0]` |
| 1D | `'right'` | `x[-1]` |
| 2D | `'left'` | `i=0` (all j) |
| 2D | `'right'` | `i=-1` (all j) |
| 2D | `'bottom'` | `j=0` (all i) |
| 2D | `'top'` | `j=-1` (all i) |

#### Custom mask BCs (2D)

```python
mask = np.zeros((nx, ny), dtype=bool)
mask[0, :ny//2] = True                       # bottom half of left wall
eq.set_bc(mask=mask, kind='dirichlet', value=2.0)
```

---

### Initial Conditions

Required for transient problems; not used by `solve_steady()`.

```python
eq.set_ic(0.0)                             # scalar — uniform field
eq.set_ic(np.sin(np.pi * x))              # array
eq.set_ic(lambda x: np.sin(np.pi * x))   # 1D callable
eq.set_ic(lambda x, y: np.sin(np.pi * x) * np.sin(np.pi * y))  # 2D callable
```

Dirichlet boundary nodes are automatically snapped to their BC values at `t=t0`,
so the IC does not need to be consistent with the BCs.

---

### Coupled Systems

Field names in callables create couplings. Declare all equations together in a
`PDESystem`:

```python
x  = np.linspace(0, 1, 256)
k1 = 10.0

eqA = PDE('cA', x=x)
eqA.add_diffusion(diffusivity=0.01)
eqA.add_source(expr=lambda x, cA, cB: -k1 * cA * cB)
eqA.set_bc(side='left',  kind='dirichlet', value=5.0)
eqA.set_bc(side='right', kind='dirichlet', value=0.0)
eqA.set_ic(0.0)

eqB = PDE('cB', x=x)
eqB.add_diffusion(diffusivity=0.01)
eqB.add_source(expr=lambda x, cA, cB: -k1 * cA * cB)
eqB.set_bc(side='left',  kind='dirichlet', value=0.0)
eqB.set_bc(side='right', kind='dirichlet', value=5.0)
eqB.set_ic(0.0)

# Transient
sol = PDESystem([eqA, eqB]).solve((0, 0.5), method='RK45')
# sol.cA  shape (256, nt)
# sol.cB  shape (256, nt)

# Steady-state — no ICs needed, provide an initial guess instead
sol = PDESystem([eqA, eqB]).solve_steady(
    guess={'cA': np.linspace(5, 0, 256),
           'cB': np.linspace(0, 5, 256)}
)
# sol.cA  shape (256,)
# sol.cB  shape (256,)
```

`PDESystem` validates field references at construction time — a typo in a field name
raises immediately before any solve.

---

### 2D Problems

Pass `y=` to `PDE` to activate 2D mode. Everything else — terms, BCs, ICs — works
identically.

```python
x = np.linspace(0, 1, 64)
y = np.linspace(0, 1, 64)

eq = PDE('T', x=x, y=y)
eq.add_diffusion(diffusivity=1.0)
eq.set_bc(side='left',   kind='dirichlet', value=1.0)
eq.set_bc(side='right',  kind='dirichlet', value=0.0)
eq.set_bc(side='bottom', kind='neumann',   value=0.0)
eq.set_bc(side='top',    kind='neumann',   value=0.0)

# Transient
eq.set_ic(0.0)
sol = eq.solve((0, 1.0), method='BDF')
# sol.T  shape (64, 64, nt)

# Steady-state
sol = eq.solve_steady()
# sol.T  shape (64, 64)
```

In 2D, `x` and `y` are passed as 2D meshgrid arrays `(nx, ny)` to callables:

```python
eq.add_diffusion(diffusivity=lambda x, y, T: 1 + 0.5 * np.sin(np.pi * x) * T)
```

---

### Interior Boundary Conditions

Mark interior cells as obstacles or inclusions:

```python
# Heated cylinder
mask = (X - 0.5)**2 + (Y - 0.5)**2 < 0.1**2
eq.set_interior_bc(mask, kind='dirichlet', value=2.0)

# Insulating obstacle (frozen at ambient value)
eq.set_interior_bc(mask, kind='neumann')
```

Interior BCs are applied after all stencil terms at every RHS evaluation.
Multiple obstacles can be registered on the same field with separate calls.

---

### Choosing a Transient Solver

Pass `method=` to `solve()` or `PDESystem.solve()`. These are forwarded directly
to `scipy.integrate.solve_ivp`.

| Problem type | Recommended method |
|---|---|
| Advection-dominated, smooth 1D | `'RK45'` |
| Diffusion-dominated, stiff | `'BDF'` |
| Stiff reaction-diffusion (Gray-Scott, FitzHugh-Nagumo) | `'BDF'` |
| High accuracy needed | `'Radau'` |
| NS cylinder Re ~ 100 | `'RK45'` |

Tight tolerances are required to resolve diffusion accurately:
`rtol=1e-8, atol=1e-10` is a safe starting point.

---

### Steady-State Problems

`solve_steady()` finds φ such that RHS(φ) = 0 by delegating to
`scipy.optimize.root`. Linear and nonlinear problems are handled identically —
Newton converges in a single iteration for linear systems, so there is no
performance penalty for always using the nonlinear path.

#### Single equation

```python
eq = PDE('T', x=x)
eq.add_diffusion(diffusivity=1.0)
eq.set_bc(side='left',  kind='dirichlet', value=1.0)
eq.set_bc(side='right', kind='dirichlet', value=0.0)

sol = eq.solve_steady()            # linear — zero guess is fine
sol = eq.solve_steady(             # nonlinear — provide a starting guess
    guess=np.linspace(1.0, 0.0, nx)
)
```

#### Coupled system

```python
sol = PDESystem([eqA, eqB, eqC]).solve_steady(
    guess={'cA': np.linspace(5, 0, nx),
           'cB': np.linspace(0, 5, nx),
           'cC': np.zeros(nx)},
)
# sol.cA, sol.cB, sol.cC  — each shape (nx,)
```

#### Parameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `guess` | `None` → zeros | Array, scalar, or dict of arrays per field |
| `method` | `'hybr'` | Any `scipy.optimize.root` method: `'hybr'`, `'lm'`, `'krylov'`, … |
| `**kwargs` | — | Forwarded to `scipy.optimize.root` (e.g. `tol`, `options`) |

#### Checking convergence

```python
sol = eq.solve_steady()
print(sol.residual)   # max|RHS(φ)| — should be < 1e-8 for a good solution
print(sol.nfev)       # number of RHS evaluations
```

Use `sol.residual` rather than `sol.success` as the primary convergence check —
see the note in [SteadySolution](#steadysolution).

---

## Pre-built Equations

All factories return configured `PDE` or `NamedPDESystem` objects. The caller sets
boundary conditions, initial conditions (transient only), and calls `solve()` or
`solve_steady()`.

### Single-field factories (return `PDE`)

All support 1D and 2D (pass `y=` for 2D), and both `solve()` and `solve_steady()`.

#### HeatEquation

```
∂T/∂t = ∇·(D ∇T)
```

```python
from upde import HeatEquation
eq = HeatEquation('T', x=x, diffusivity=0.01)
```

#### AdvectionDiffusion

```
∂φ/∂t + c·∇φ = ∇·(D ∇φ)
```

```python
from upde import AdvectionDiffusion
eq = AdvectionDiffusion('phi', x=x, velocity=1.0, diffusivity=0.01)
# 2D
eq = AdvectionDiffusion('phi', x=x, y=y, velocity_x=1.0, velocity_y=0.5, diffusivity=0.01)
```

#### Burgers

```
∂u/∂t + ∂(u²/2)/∂x = ν ∂²u/∂x²
```

```python
from upde import Burgers
eq = Burgers('u', x=x, viscosity=0.01)   # 1D only
```

#### ConservationLaw

```
∂u/∂t + ∂F(u)/∂x = 0
```

```python
from upde import ConservationLaw
eq = ConservationLaw('u', x=x, flux=lambda u: 0.5 * u**2)
```

#### ReactionDiffusion

```
∂u/∂t = ∇·(D ∇u) + R(u, ...)
```

```python
from upde import ReactionDiffusion
eq = ReactionDiffusion('u', x=x, diffusivity=0.01, reaction=lambda x, u: -u * (1 - u))
```

#### MixtureFraction

```
∂Z/∂t + u·∇Z = ∇·(D ∇Z)
```

```python
from upde import MixtureFraction
eq = MixtureFraction('Z', x=x, velocity=0.1, diffusivity=1e-4)
```

---

### Multi-field factories (return `NamedPDESystem`)

These expose each equation as a named attribute for convenient BC/IC setup.

#### WaveEquation

```
∂²u/∂t² = c² ∇²u     [split: ∂u/∂t = uₜ,  ∂uₜ/∂t = c² ∇²u]
```

```python
from upde import WaveEquation
ns = WaveEquation('u', 'ut', x=x, speed=1.0)
ns.u.set_bc(kind='periodic')
ns.ut.set_bc(kind='periodic')
ns.u.set_ic(np.sin(2 * np.pi * x))
ns.ut.set_ic(np.zeros_like(x))
sol = ns.solve((0, 2.0))
# sol.u, sol.ut
```

#### GrayScott

```
∂u/∂t = Dᵤ ∇²u − uv² + F(1−u)
∂v/∂t = D_v ∇²v + uv² − (F+k)v
```

```python
from upde import GrayScott
gs = GrayScott('u', 'v', x=x, y=y, Du=2e-5, Dv=1e-5, F=0.04, k=0.06)
gs.u.set_bc(kind='periodic', side='all')
gs.v.set_bc(kind='periodic', side='all')
gs.u.set_ic(1.0)
gs.v.set_ic(lambda x, y: np.where((x-0.5)**2 + (y-0.5)**2 < 0.01, 0.25, 0.0))
sol = gs.solve((0, 5000), method='BDF')
```

#### NavierStokes2D

2D incompressible Navier-Stokes via artificial compressibility (Chorin 1967):

```
∂u/∂t = −u·∇u − (1/ρ) ∂p/∂x + ν ∇²u
∂v/∂t = −u·∇v − (1/ρ) ∂p/∂y + ν ∇²v
∂p/∂t = −β ∇·u
```

```python
from upde import NavierStokes2D
ns = NavierStokes2D('u', 'v', 'p', x=x, y=y, nu=0.01, rho=1.0, beta=10.0)
# Set BCs on ns.u, ns.v, ns.p ...
sol = ns.solve((0, 10.0), method='RK45')
# sol.u, sol.v, sol.p  each shape (nx, ny, nt)
```

`pressure_stabilisation` (default `None` → `0.5 Δx²`) suppresses checkerboard
pressure oscillations on the collocated grid. Set to `0.0` to disable.

---

## Combustion and Flamelet Chemistry

uPDE supports flamelet-based combustion modelling via mixture-fraction transport
coupled to a precomputed chemistry table. This approach avoids the extreme stiffness
of full chemistry by decoupling the transport solve from the thermochemistry lookup.

### FlameletTable

```python
from upde.chemistry import FlameletTable
```

Maps Z ∈ [0, 1] to temperature and species mass fractions via `numpy.interp`.

#### Construction

```python
# Analytic Burke-Schumann (no dependencies — good for testing)
table = FlameletTable.burke_schumann(
    Z_st=0.055, T_fuel=300.0, T_ox=300.0, T_ad=2230.0
)

# Load a previously saved table
table = FlameletTable.from_file('ch4_air.npz')

# From raw arrays
table = FlameletTable(Z_grid, T, species={'CH4': Y_CH4, 'O2': Y_O2, ...})

# From a Cantera counterflow flame (requires pip install cantera)
table = FlameletTable.from_cantera(
    mechanism='gri30.yaml', fuel='CH4', oxidizer='O2:0.21,N2:0.79',
    T_fuel=300.0, T_ox=300.0,
)
table.save('ch4_air_gri30.npz')
```

#### Accessors

```python
T_field   = table.T(Z)                   # temperature [K]
Y_CH4     = table.Y('CH4', Z)            # species mass fraction
rho_field = table.rho(Z, P=101325.0)     # density [kg/m³]

print(table.species)   # ['CH4', 'O2', 'CO2', 'H2O', 'N2']
print(table.Z_st)      # stoichiometric mixture fraction
```

All accessors accept any NumPy array shape. Z is clipped to [0, 1] automatically.

### Workflow

```python
import numpy as np
from upde import MixtureFraction, PDESystem
from upde.chemistry import FlameletTable

table = FlameletTable.burke_schumann()

x  = np.linspace(0, 1, 256)
eq = MixtureFraction('Z', x=x, diffusivity=1e-4)
eq.set_bc(side='left',  kind='dirichlet', value=0.0)
eq.set_bc(side='right', kind='dirichlet', value=1.0)
eq.set_ic(0.0)

sol = PDESystem([eq]).solve(t_span=(0, 5000), method='BDF', rtol=1e-6, atol=1e-8)

Z_final = sol.Z[:, -1]
T       = table.T(Z_final)
Y_CH4   = table.Y('CH4', Z_final)

print(f'Flame at x = {x[np.argmax(T)]:.4f}  (Z_st = {table.Z_st:.4f})')
```

---

## API Reference

### PDE

```python
PDE(field, x, y=None)
```

| Method | Description |
|--------|-------------|
| `add_advection(velocity=)` | 1D convective form $-c\,\partial_x\phi$, upwind on sign(c) |
| `add_advection(velocity_x=, velocity_y=)` | 2D convective form |
| `add_diffusion(diffusivity=)` | Isotropic diffusion $\nabla\cdot(D\nabla\phi)$ |
| `add_diffusion(diffusivity_x=, diffusivity_y=)` | Anisotropic diffusion |
| `add_source(expr=)` | Pointwise source $S(x[,y], \texttt{**fields})$ |
| `add_flux(flux=, scheme=)` | 1D conservation law $-\partial_x F(\phi)$ |
| `add_flux(flux_x=, flux_y=, scheme=)` | 2D conservation law |
| `add_term(fn)` | Generic operator term — see [Operator Reference](#operator-reference) |
| `set_bc(kind, side=, mask=, value=)` | Domain boundary condition |
| `set_interior_bc(mask, kind, value=)` | Interior obstacle BC |
| `set_ic(ic)` | Initial condition: scalar, array, or callable |
| `solve(t_span, **kwargs)` | Transient solve (uncoupled equations only) |
| `solve_steady(guess=None, method='hybr', **kwargs)` | Steady-state solve (uncoupled only) |
| `field_refs()` | Set of external field names referenced by this equation |

---

### PDESystem

```python
PDESystem(equations)
```

Validates field references at construction time.

| Method | Description |
|--------|-------------|
| `solve(t_span, ICs=None, method='RK45', t_eval=None, **kwargs)` | Transient solve via `solve_ivp` |
| `solve_steady(guess=None, method='hybr', **kwargs)` | Steady-state solve via `root` |

`**kwargs` are forwarded to the underlying SciPy solver.

---

### PDESolution

Returned by `solve()`.

| Attribute | Shape | Description |
|-----------|-------|-------------|
| `sol.<field>` | `(nx, nt)` or `(nx, ny, nt)` | Field values, spatial indices first |
| `sol.t` | `(nt,)` | Time points |
| `sol.success` | bool | `True` if integrator reached `t_span[1]` |
| `sol.message` | str | Integrator status |
| `sol.raw` | `OdeResult` | Raw SciPy result |

---

### SteadySolution

Returned by `solve_steady()`.

| Attribute | Shape | Description |
|-----------|-------|-------------|
| `sol.<field>` | `(nx,)` or `(nx, ny)` | Field values — no time dimension |
| `sol.residual` | float | max\|RHS(φ)\| at solution — primary convergence indicator |
| `sol.success` | bool | Convergence flag from `root` (see note above) |
| `sol.message` | str | Convergence message |
| `sol.nfev` | int | Number of RHS evaluations |
| `sol.raw` | `OptimizeResult` | Raw SciPy result |

---

### Operator Reference

Operators are injected into `add_term` functions by **name matching** — the order of
arguments does not matter:

```python
def my_rhs(u, Dx, Dxx):   ...   # order is irrelevant — only names matter
def my_rhs(Dxx, u, Dx):   ...   # equivalent

# Use **fields to receive all coupled fields at once
def my_rhs(Dx, Dy, **fields):
    u, v = fields['u'], fields['v']
    ...
```

| Name | Expression | Notes |
|------|-----------|-------|
| `Dx(phi)` | $\partial\phi/\partial x$ | 2nd-order central |
| `Dx(phi, 'upwind', c=v)` | $\partial\phi/\partial x$ | 1st-order upwind on sign(v) |
| `Dy(phi)` | $\partial\phi/\partial y$ | 2D only |
| `Dy(phi, 'upwind', c=v)` | $\partial\phi/\partial y$ | 2D only |
| `Dxx(phi)` | $\partial^2\phi/\partial x^2$ | 2nd-order central |
| `Dyy(phi)` | $\partial^2\phi/\partial y^2$ | 2D only |
| `Div_x(c, phi)` | $-\partial(c\phi)/\partial x$ | Central conservative |
| `Div_y(c, phi)` | $-\partial(c\phi)/\partial y$ | 2D only |
| `Div_flux_x(k, phi)` | $\partial(k\,\partial\phi/\partial x)/\partial x$ | Diffusive flux |
| `Div_flux_y(k, phi)` | $\partial(k\,\partial\phi/\partial y)/\partial y$ | 2D only |

> **Rule:** never nest operators — use `Dxx(phi)`, not `Dx(Dx(phi))`.

---

## Numerical Methods

| Term | Scheme | Order |
|------|--------|-------|
| `add_advection` | Convective upwind | 1st in space |
| `add_diffusion` | Conservative central | 2nd in space |
| `add_flux` (upwind) | Donor-cell, wave speed inferred | 1st in space |
| `add_flux` (central) | Central flux divergence | 2nd in space |
| `Dx`, `Dy` | Central difference | 2nd in space |
| `Dxx`, `Dyy` | Central difference | 2nd in space |
| `Div_x`, `Div_y` | Central conservative | 2nd in space |
| Transient time integration | `solve_ivp` adaptive | depends on method |
| Steady-state solve | Newton via `scipy.optimize.root` | — |

All spatial operators are BC-aware — ghost cells are padded from the boundary
conditions before each stencil application, preventing periodic wrap-around
from silently corrupting non-periodic boundaries.

---

## Troubleshooting

**Transient solver fails or takes tiny steps**
- Switch to `method='BDF'` for diffusion-dominated or stiff problems
- Loosen tolerances: `rtol=1e-3, atol=1e-5`
- Check that BCs are consistent with the IC at `t=0`

**Solution blows up (transient)**
- Add or increase diffusivity to stabilise advection
- Reduce `beta` in NavierStokes2D if the pressure equation is oscillating
- Check that source terms have the correct sign

**Checkerboard oscillations in pressure (NavierStokes2D)**
- The default `pressure_stabilisation=None` sets ε = 0.5 Δx² automatically
- If oscillations persist, increase it: `pressure_stabilisation=dx**2`

**Steady-state solver doesn't converge**
- Check `sol.residual` — if it's small (< 1e-6) the solution is likely correct despite `sol.success=False`
- For nonlinear problems, provide a better `guess` (e.g. linear interpolation between BCs)
- Try a different method: `solve_steady(method='lm')` or `solve_steady(method='krylov')`
- For large 2D problems, pass `options={'maxfev': 10000}` to increase the iteration limit

**`ValueError: references external field(s)`**
- `PDE.solve()` and `PDE.solve_steady()` only work for uncoupled equations
- Use `PDESystem([eq1, eq2, ...]).solve(...)` or `.solve_steady(...)` for coupled problems

**Field not found in solution**
- Check the field name string matches exactly — names are case-sensitive
- `'t'` is reserved for the solver time — `PDE('t', ...)` raises `ValueError`
