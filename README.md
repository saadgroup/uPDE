# uPDE

> A lightweight Python library for solving 1D and 2D PDEs using the method of lines.

uPDE lets you describe a PDE — its terms, coefficients, boundary conditions, and initial data — and hands the spatial right-hand side directly to `scipy.integrate.solve_ivp`. You get adaptive step control, stiffness detection, and a choice of explicit and implicit integrators without writing a single line of time-stepping code.

**Features:**
- 1D and 2D problems on uniform Cartesian grids
- Coupled multi-field systems
- Scalar, array, and callable (nonlinear/field-dependent) coefficients
- Dirichlet, Neumann, and per-axis periodic boundary conditions
- Pure NumPy/SciPy — no compilation, no external mesh libraries

---

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Core Concepts](#core-concepts)
- [User Guide](#user-guide)
  - [Coefficients and Callables](#coefficients-and-callables)
  - [Boundary Conditions](#boundary-conditions)
  - [Initial Conditions](#initial-conditions)
  - [Coupled Systems](#coupled-systems)
  - [2D Problems](#2d-problems)
  - [Choosing a Solver](#choosing-a-solver)
- [API Reference](#api-reference)
  - [PDE](#pde)
  - [PDESystem](#pdesystem)
  - [PDESolution](#pdesolution)
- [Examples](#examples)
  - [Linear Advection](#linear-advection)
  - [Viscous Burgers Equation](#viscous-burgers-equation)
  - [Reaction-Diffusion A + B → C](#reaction-diffusion-a--b--c)
  - [2D Heat Equation](#2d-heat-equation)
  - [2D Nonlinear Diffusion](#2d-nonlinear-diffusion)
  - [2D Custom Mask BC](#2d-custom-mask-bc)
- [Numerical Methods](#numerical-methods)
- [Troubleshooting](#troubleshooting)

---

## Installation

uPDE has no compiled dependencies. Copy `pde_solver.py` into your project, or install from the repository:

```bash
# from source
pip install git+https://github.com/YOURUSERNAME/upde.git

# or just copy the file
cp pde_solver.py your_project/
```

**Requirements:**

| Package | Version | Notes |
|---------|---------|-------|
| numpy   | ≥ 1.21  | Array operations, stencil construction |
| scipy   | ≥ 1.7   | `solve_ivp` time integration |
| python  | ≥ 3.9   | Required for `inspect.signature` features |

---

## Quick Start

```python
import numpy as np
from pde_solver import PDE, PDESystem

x = np.linspace(0, 1, 256)

eq = PDE('T', x=x)
eq.add_diffusion(diffusivity=0.05)
eq.set_bc(side='left',  kind='dirichlet', value=1.0)
eq.set_bc(side='right', kind='dirichlet', value=0.0)
eq.set_ic(0.0)

sol = PDESystem([eq]).solve(t_span=(0, 1), method='RK45')

# sol.T         shape (256, nt)
# sol.T[:, -1]  final profile
# sol.t         time points
```

---

## Core Concepts

### PDE — equation descriptor

A `PDE` object holds the mathematical description of one equation: its field name, spatial grid, terms (advection, diffusion, source), boundary conditions, and initial condition. It never solves anything on its own.

### PDESystem — the solver

`PDESystem` takes a list of `PDE` objects, validates that all field references are consistent, assembles the joint state vector, and calls `solve_ivp`. Even single-equation problems use `PDESystem([eq]).solve(...)`.

> **Why no `PDE.solve()`?**
> In a coupled system, the advection velocity of field *T* may be the current value of field *u*, which the *T*-equation does not own. Solving requires the complete joint state — only `PDESystem` has that. Calling `solve()` on a single `PDE` would silently treat missing coupled fields as zero, which is almost never what you want.

### PDESolution — the result

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

---

## User Guide

### Coefficients and Callables

Every coefficient in uPDE — diffusivity, velocity, source — accepts three forms:

| Form | Example | Behaviour |
|------|---------|-----------|
| scalar | `0.05` | Broadcast to a constant array over the whole grid |
| ndarray | `D_arr` | Used as-is; must match the field shape |
| callable (1D) | `lambda x, T: 1 + T**2` | Called as `f(x, **fields)` at every time step |
| callable (2D) | `lambda x, y, T: x * T` | Called as `f(x, y, **fields)`; `x`, `y` are meshgrid arrays |

uPDE uses `inspect.signature` to forward only the arguments a callable actually declares — a coefficient that depends only on `x` will not receive field arrays. Extra physical parameters go in via closures:

```python
k1 = 10.0
# k1 is captured from the enclosing scope — uPDE never sees it
eq.add_source(expr=lambda x, cA, cB: -k1 * cA * cB)
```

---

### Boundary Conditions

#### Dirichlet — prescribed value

```python
eq.set_bc(side='left',  kind='dirichlet', value=1.0)
eq.set_bc(side='right', kind='dirichlet', value=0.0)

# time-varying: value is a callable of t
eq.set_bc(side='left', kind='dirichlet', value=lambda t: np.sin(t))
```

#### Neumann — prescribed normal derivative

```python
# zero-flux (insulating wall)
eq.set_bc(side='right', kind='neumann', value=0.0)

# prescribed heat flux
eq.set_bc(side='left',  kind='neumann', value=2.5)
```

#### Periodic

```python
# 1D
eq.set_bc(kind='periodic')

# 2D — must specify which axis
eq.set_bc(kind='periodic', side='x')    # wrap left ↔ right
eq.set_bc(kind='periodic', side='y')    # wrap bottom ↔ top
eq.set_bc(kind='periodic', side='all')  # fully periodic
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

For non-standard regions — e.g. two heating strips on the same wall — pass a boolean array:

```python
hot  = np.zeros((nx, ny), dtype=bool)
cold = np.zeros((nx, ny), dtype=bool)

hot[0, :ny//3]           = True   # bottom third of left wall
hot[0, 2*ny//3:]         = True   # top third of left wall
cold[0, ny//3:2*ny//3]  = True   # middle third

eq.set_bc(mask=hot,  kind='dirichlet', value=2.0)
eq.set_bc(mask=cold, kind='dirichlet', value=0.0)
```

---

### Initial Conditions

```python
eq.set_ic(0.0)                                          # scalar — constant field
eq.set_ic(np.zeros(n))                                  # 1D array
eq.set_ic(lambda x: np.sin(np.pi * x))                 # 1D callable
eq.set_ic(lambda x, y: np.sin(np.pi*x)*np.sin(np.pi*y)) # 2D callable

# 2D array IC — step function at x = L/2
U0 = np.zeros((nx, ny))
U0[:, nx//2:] = 1.0
eq.set_ic(U0)

# ICs can be overridden at solve time (useful for parameter sweeps)
sol = PDESystem([eq]).solve(t_span=(0, 1), ICs={'T': hot_ic})
```

Dirichlet boundary points are automatically snapped to their prescribed values before the first time step.

---

### Coupled Systems

Multi-field problems are built by creating one `PDE` per field and passing them all to `PDESystem`. Fields are coupled by referencing each other's names in callable coefficients. `PDESystem` validates all references at construction time — a typo in a field name raises a `ValueError` immediately.

```python
k1 = 10.0

eqA = PDE('cA', x=x)
eqA.add_diffusion(diffusivity=1.0)
eqA.add_source(expr=lambda x, cA, cB: -k1 * cA * cB)

eqB = PDE('cB', x=x)
eqB.add_diffusion(diffusivity=1.0)
eqB.add_source(expr=lambda x, cA, cB: -k1 * cA * cB)

sol = PDESystem([eqA, eqB]).solve(t_span=(0, 0.5), method='RK45')
# sol.cA, sol.cB — each shape (n, nt)
```

---

### 2D Problems

Pass a second grid `y` to the `PDE` constructor. Everything else — terms, BCs, ICs — uses the same API as 1D.

```python
x = np.linspace(0, 1, 64)
y = np.linspace(0, 1, 64)

eq = PDE('T', x=x, y=y)             # 2D mode activated

eq.add_diffusion(diffusivity=0.05)   # isotropic

eq.add_diffusion(diffusivity_x=0.1,  # anisotropic
                 diffusivity_y=0.005)

eq.add_advection(velocity_x=1.0, velocity_y=0.5)  # 2D advection

eq.add_diffusion(diffusivity=lambda x, y, T: 1 + T**2)  # nonlinear

# sol.T has shape (64, 64, nt)
# sol.T[:, :, -1]   final frame
# sol.T[:, 32, -1]  profile at y = 0.5
```

> **Stiffness warning:** 2D diffusion is always stiff — the explicit stability limit scales as Δt ∼ Δx². On a 64×64 grid with D=0.05 this is roughly 1.9×10⁻³. Use `method='BDF'` for any diffusion-dominated 2D problem.

---

### Choosing a Solver

All solvers are passed as `method=` to `PDESystem.solve()`.

| Method | Type | Best for |
|--------|------|----------|
| `'RK45'` | Explicit | Advection-dominated problems, 1D diffusion, default choice |
| `'RK23'` | Explicit | Low-accuracy or quick exploratory runs |
| `'BDF'` | Implicit | 2D diffusion, stiff reaction-diffusion — **recommended for stiff problems** |
| `'Radau'` | Implicit | Very stiff systems; more accurate than BDF but slower per step |
| `'DOP853'` | Explicit | Smooth non-stiff problems where high accuracy is needed |

```python
sol = PDESystem([eq]).solve(
    t_span=(0, 1),
    method='BDF',
    rtol=1e-4,
    atol=1e-6,
    t_eval=np.linspace(0, 1, 50)   # request specific output times
)
```

---

## API Reference

### PDE

```python
PDE(field, x, y=None)
```

Descriptor for a single field equation. Accumulates terms, BCs, and IC without integrating.

| Parameter | Type | Description |
|-----------|------|-------------|
| `field` | `str` | Name of the unknown, e.g. `'T'` or `'cA'` |
| `x` | `ndarray (nx,)` | 1D coordinate array in x |
| `y` | `ndarray (ny,)` or `None` | 1D coordinate array in y. Passing this activates 2D mode. |

---

#### `.add_advection(velocity=)` &nbsp;·&nbsp; `.add_advection(velocity_x=, velocity_y=)`

Adds an advection term. All term-building methods return `self` for chaining.

- **1D:** `−∂(c·φ)/∂x`
- **2D:** `−∂(cx·φ)/∂x − ∂(cy·φ)/∂y`

| Parameter | Type | Description |
|-----------|------|-------------|
| `velocity` | scalar \| ndarray \| callable | **1D only.** Advection speed `c(x, **fields)` |
| `velocity_x` | scalar \| ndarray \| callable | **2D only.** x-component of velocity |
| `velocity_y` | scalar \| ndarray \| callable | **2D only.** y-component of velocity |

---

#### `.add_diffusion(diffusivity=)` &nbsp;·&nbsp; `.add_diffusion(diffusivity_x=, diffusivity_y=)`

Adds a diffusion term using conservative central differences with face-averaged diffusivities.

- **1D:** `∂(D·∂φ/∂x)/∂x`
- **2D isotropic:** `∂(D·∂φ/∂x)/∂x + ∂(D·∂φ/∂y)/∂y`
- **2D anisotropic:** `∂(Dx·∂φ/∂x)/∂x + ∂(Dy·∂φ/∂y)/∂y`

| Parameter | Type | Description |
|-----------|------|-------------|
| `diffusivity` | scalar \| ndarray \| callable | Isotropic diffusivity `D(x[,y], **fields)` |
| `diffusivity_x` | scalar \| ndarray \| callable | **2D anisotropic.** Diffusivity in x direction |
| `diffusivity_y` | scalar \| ndarray \| callable | **2D anisotropic.** Diffusivity in y direction |

---

#### `.add_source(expr)`

Adds a pointwise source term added directly to the RHS. May reference other fields.

| Parameter | Type | Description |
|-----------|------|-------------|
| `expr` | scalar \| ndarray \| callable | Source `S(x[,y], **fields)` |

---

#### `.add_flux(flux, scheme='upwind')` &nbsp;·&nbsp; *1D only*

Escape hatch for a custom precomputed flux F. Adds `−∂F/∂x`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `flux` | callable | `F(x, **fields)` → ndarray |
| `scheme` | `'upwind'` \| `'central'` | Differencing scheme for `−∂F/∂x` |

---

#### `.set_bc(kind, side=None, mask=None, value=None)`

Sets a boundary condition. Multiple calls accumulate — they do not overwrite each other.

| Parameter | Type | Description |
|-----------|------|-------------|
| `kind` | `'dirichlet'` \| `'neumann'` \| `'periodic'` | BC type |
| `side` | `str` or `None` | Named edge shorthand. See [Side shortcuts](#side-shortcuts). |
| `mask` | `bool ndarray` or `None` | Explicit boolean mask matching the field shape. Overrides `side=`. |
| `value` | scalar \| `callable(t)` | Prescribed value (Dirichlet) or outward normal derivative (Neumann). Pass a callable for time-varying BCs. |

---

#### `.set_ic(ic)`

Sets the initial condition.

| Parameter | Type | Description |
|-----------|------|-------------|
| `ic` | scalar \| ndarray \| callable | Scalar: broadcast to whole grid. Array: shape `(nx,)` or `(nx, ny)`. Callable 1D: `ic(x)`. Callable 2D: `ic(x, y)` with meshgrid arrays. |

---

### PDESystem

```python
PDESystem(equations)
```

Couples a list of PDE descriptors. Validates at construction time: no duplicate field names, consistent grid sizes, all callable field references declared.

| Parameter | Type | Description |
|-----------|------|-------------|
| `equations` | `list[PDE]` | One or more PDE descriptors |

---

#### `.solve(t_span, ICs=None, method='RK45', t_eval=None, **kwargs)`

Integrates the system. All extra keyword arguments are forwarded to `scipy.integrate.solve_ivp`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `t_span` | `(float, float)` | Start and end time `(t0, tf)` |
| `ICs` | `dict` or `None` | Override ICs per field: `{'T': array_or_callable}`. Takes priority over `eq.set_ic()`. |
| `method` | `str` | Integrator. See [Choosing a Solver](#choosing-a-solver). |
| `t_eval` | `ndarray` or `None` | Times at which to store output. If `None`, the integrator chooses its own steps. |
| `rtol`, `atol` | `float` | Relative and absolute tolerances forwarded to `solve_ivp`. |
| `max_step` | `float` | Maximum step size forwarded to `solve_ivp`. |

**IC priority order:** `ICs` dict argument → `eq.set_ic()` → `ValueError`

---

### PDESolution

Returned by `PDESystem.solve()`.

| Attribute | Type | Description |
|-----------|------|-------------|
| `sol.<field>` | ndarray | Solution for each field by name. Shape `(nx, nt)` in 1D, `(nx, ny, nt)` in 2D. |
| `sol.t` | `ndarray (nt,)` | Time points at which solution was stored |
| `sol.success` | `bool` | `True` if the integrator converged |
| `sol.message` | `str` | Human-readable integrator status |
| `sol.raw` | `OdeResult` | Full scipy `solve_ivp` result (dense output, events, step history) |

---

## Examples

### Linear Advection

A Gaussian pulse at speed c=1 with periodic BCs — the pulse should return to its starting position at t = 2π.

```python
x  = np.linspace(0, 2*np.pi, 256)
ic = np.exp(-((x - np.pi/3)**2) / 0.05)

eq = PDE('phi', x=x)
eq.add_advection(velocity=1.0)
eq.set_bc(kind='periodic')
eq.set_ic(ic)

sol = PDESystem([eq]).solve(
    t_span=(0, 6),
    method='RK23',
    t_eval=np.linspace(0, 6, 120))
```

---

### Viscous Burgers Equation

Nonlinear advection with self-advection velocity c = u. The self-steepening term forms a near-shock which the small viscosity regularises.

```python
x = np.linspace(0, 2*np.pi, 512)

eq = PDE('u', x=x)
eq.add_advection(velocity=lambda x, u: u)   # c = u(x, t)
eq.add_diffusion(diffusivity=0.005)
eq.set_bc(kind='periodic')
eq.set_ic(np.sin(x) + 0.5*np.sin(2*x))

sol = PDESystem([eq]).solve(
    t_span=(0, 3),
    method='RK45',
    t_eval=np.linspace(0, 3, 100))
```

---

### Reaction-Diffusion A + B → C

Three coupled species diffusing in from opposite ends and reacting where they meet. All three equations must be solved jointly because the source terms couple them.

```python
x  = np.linspace(0, 1, 256)
k1 = 10.0

eqA = PDE('cA', x=x)
eqA.add_diffusion(diffusivity=1.0)
eqA.add_source(expr=lambda x, cA, cB, cC: -k1 * cA * cB)
eqA.set_bc(side='left',  kind='dirichlet', value=5.0)
eqA.set_bc(side='right', kind='dirichlet', value=0.0)
eqA.set_ic(0.0)

eqB = PDE('cB', x=x)
eqB.add_diffusion(diffusivity=1.0)
eqB.add_source(expr=lambda x, cA, cB, cC: -k1 * cA * cB)
eqB.set_bc(side='left',  kind='dirichlet', value=0.0)
eqB.set_bc(side='right', kind='dirichlet', value=3.0)
eqB.set_ic(0.0)

eqC = PDE('cC', x=x)
eqC.add_diffusion(diffusivity=1.0)
eqC.add_source(expr=lambda x, cA, cB, cC: +k1 * cA * cB)
eqC.set_bc(side='left',  kind='dirichlet', value=0.0)
eqC.set_bc(side='right', kind='dirichlet', value=0.0)
eqC.set_ic(0.0)

sol = PDESystem([eqA, eqB, eqC]).solve(
    t_span=(0, 0.5), method='RK45', rtol=1e-6, atol=1e-8)

# sol.cA, sol.cB, sol.cC — each shape (256, nt)
```

---

### 2D Heat Equation

Constant diffusivity on a unit square, Dirichlet left/right walls, insulating top/bottom. Steady state: T = 1 − x.

```python
x = np.linspace(0, 1, 64)
y = np.linspace(0, 1, 64)

eq = PDE('T', x=x, y=y)
eq.add_diffusion(diffusivity=0.05)
eq.set_bc(side='left',   kind='dirichlet', value=1.0)
eq.set_bc(side='right',  kind='dirichlet', value=0.0)
eq.set_bc(side='bottom', kind='neumann',   value=0.0)
eq.set_bc(side='top',    kind='neumann',   value=0.0)
eq.set_ic(0.0)

sol = PDESystem([eq]).solve(
    t_span=(0, 1),
    method='BDF',          # BDF required — 2D diffusion is stiff
    rtol=1e-4, atol=1e-6,
    t_eval=np.linspace(0, 1, 10))

# sol.T.shape == (64, 64, 10)
# sol.T[:, :, -1]   final 2D field
# sol.T[:, 32, -1]  profile at y = 0.5
```

---

### 2D Nonlinear Diffusion

Field-dependent diffusivity κ(T) = 1 + T² — regions with high T diffuse faster.

```python
x = np.linspace(0, 1, 48)
y = np.linspace(0, 1, 48)

eq = PDE('T', x=x, y=y)
eq.add_diffusion(diffusivity=lambda x, y, T: 1.0 + T**2)
eq.set_bc(side='left',   kind='dirichlet', value=2.0)
eq.set_bc(side='right',  kind='dirichlet', value=0.0)
eq.set_bc(side='bottom', kind='neumann',   value=0.0)
eq.set_bc(side='top',    kind='neumann',   value=0.0)
eq.set_ic(0.0)

sol = PDESystem([eq]).solve(
    t_span=(0, 0.5), method='BDF', rtol=1e-4, atol=1e-6)
```

---

### 2D Custom Mask BC

Two heating strips at different temperatures on the left wall, with all remaining edges insulating.

```python
nx, ny = 64, 64
x = np.linspace(0, 1, nx)
y = np.linspace(0, 1, ny)

hot  = np.zeros((nx, ny), dtype=bool)
cold = np.zeros((nx, ny), dtype=bool)

hot[0, :ny//3]          = True   # bottom third — hot
hot[0, 2*ny//3:]        = True   # top third    — hot
cold[0, ny//3:2*ny//3] = True   # middle third — cold

eq = PDE('T', x=x, y=y)
eq.add_diffusion(diffusivity=0.05)
eq.set_bc(mask=hot,  kind='dirichlet', value=2.0)
eq.set_bc(mask=cold, kind='dirichlet', value=0.0)
eq.set_bc(side='right',  kind='neumann', value=0.0)
eq.set_bc(side='bottom', kind='neumann', value=0.0)
eq.set_bc(side='top',    kind='neumann', value=0.0)
eq.set_ic(0.0)

sol = PDESystem([eq]).solve(
    t_span=(0, 2), method='BDF', rtol=1e-4, atol=1e-6)
```

---

## Numerical Methods

### Advection — first-order upwind

The advection term −∂(cφ)/∂x is discretised using a first-order upwind scheme that selects the stencil direction based on the sign of c:

```
dF/dx ≈  (F_i − F_{i−1}) / Δx    if c ≥ 0   (backward difference)
         (F_{i+1} − F_i) / Δx    if c < 0   (forward difference)
```

In 2D, x- and y-sweeps are applied independently. The 1D stencil uses `numpy.roll`; the 2D stencil uses ghost-cell padding (see below).

### Diffusion — conservative central differences

The diffusion term ∂(D·∂φ/∂x)/∂x is discretised in conservative form with face-averaged diffusivities:

```
[∂(D∂φ/∂x)/∂x]_i ≈ (D_{i+½}(φ_{i+1}−φ_i) − D_{i−½}(φ_i−φ_{i−1})) / Δx²

D_{i+½} = ½(D_i + D_{i+1})
```

Using face averages rather than nodal values ensures conservation and correctness for nonlinear D(φ).

### Ghost-cell padding for non-periodic boundaries

`numpy.roll` wraps arrays cyclically — correct for periodic BCs but dangerous otherwise. In 2D, the left wall would receive a ghost from the right wall, injecting a large spurious flux that causes immediate blow-up.

uPDE pads the array with a ghost layer before applying each stencil. For non-periodic boundaries the ghost replicates the boundary value (zero gradient). The actual BC enforcement — zeroing Dirichlet rows, correcting Neumann rows — happens in `_apply_bcs` after the stencil.

### Dirichlet BCs — time-derivative formulation

Rather than pinning boundary nodes to a fixed value, uPDE sets the RHS at Dirichlet nodes to dg/dt. For constant BCs this is zero; for time-varying g(t) it is estimated by finite difference with Δt = 10⁻⁸. This lets the ODE integrator treat all nodes uniformly.

---

## Troubleshooting

**Solution blows up immediately in 2D**
Almost certainly a stiffness issue. Switch from `method='RK45'` to `method='BDF'`. The explicit stability limit for 2D diffusion scales as Δt ∼ Δx², which becomes very restrictive on fine grids.

**`ValueError: undeclared field(s)`**
A callable references a field name not declared in the system. Check for typos — field names are case-sensitive. Every name used in a lambda (other than `x`, `y`, `t`) must match a field declared in one of the equations passed to `PDESystem`.

**`ValueError: no initial condition for field X`**
Either call `eq.set_ic(...)` on the equation, or pass `ICs={'X': value}` to `PDESystem.solve()`.

**`sol.success` is `False`**
Check `sol.message` for the integrator's error. Common causes: tolerances too tight, step-size limit hit (try relaxing `rtol`/`atol`), or a genuinely ill-posed problem (negative diffusivity, unbounded source term).

**Neumann BC has no visible effect**
For `value=0.0`, the ghost-cell default already gives zero-flux — no correction is applied. Non-zero Neumann values are corrected automatically in `_apply_bcs`. If you see unexpected behaviour with a `mask=`-based Neumann BC, use `side=` instead so the outward normal direction is unambiguous.

**Periodic BCs in 2D cause unexpected wrapping**
`set_bc(kind='periodic')` without `side=` defaults to fully periodic (`side='all'`). For periodicity in only one direction, use `side='x'` or `side='y'` explicitly.

---

*uPDE — method of lines on top of SciPy · pure NumPy · no compilation required*
