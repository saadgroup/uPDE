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

All term-building methods return `self` for chaining.

---

#### `.add_advection(velocity=c)` · `.add_advection(velocity_x=cx, velocity_y=cy)`

Adds the **convective form** of advection: `−c · ∂φ/∂x`. `c` is treated as the characteristic
speed and sets the upwind direction via `sign(c)`. This is **not** the conservative form — use
`add_flux` when your equation is a conservation law `∂u/∂t + ∂F(u)/∂x = 0`.

- **1D:** adds `−c · ∂φ/∂x`
- **2D:** adds `−cx · ∂φ/∂x − cy · ∂φ/∂y`

```python
# constant speed
eq.add_advection(velocity=1.0)

# spatially varying speed
eq.add_advection(velocity=lambda x: np.sin(x))

# coupled field as velocity (2D)
eq_T.add_advection(velocity_x='u', velocity_y='v')
```

All velocity arguments accept: **scalar | ndarray | callable(x[,y], \*\*fields) | string field name**

---

#### `.add_diffusion(diffusivity=D)` · `.add_diffusion(diffusivity_x=Dx, diffusivity_y=Dy)`

Adds conservative diffusion `∂(D·∂φ/∂x)/∂x` with face-averaged diffusivities. Handles
constant, spatially varying, and field-dependent (nonlinear) diffusivity.

- **1D:** `∂(D·∂φ/∂x)/∂x`
- **2D isotropic:** `∂(D·∂φ/∂x)/∂x + ∂(D·∂φ/∂y)/∂y`
- **2D anisotropic:** `∂(Dx·∂φ/∂x)/∂x + ∂(Dy·∂φ/∂y)/∂y`

```python
eq.add_diffusion(diffusivity=0.01)                          # constant
eq.add_diffusion(diffusivity=lambda x, T: 1.0 + T**2)      # nonlinear D(T)
eq.add_diffusion(diffusivity_x=Dx, diffusivity_y=Dy)        # anisotropic 2D
```

---

#### `.add_source(expr=S)`

Adds a pointwise source term `S(x[,y], **fields)` directly to the RHS. Can reference any
coupled field by name.

```python
eq.add_source(expr=lambda x, cA, cB: -k * cA * cB)   # reaction sink
eq.add_source(expr=5.0)                               # uniform source
```

---

#### `.add_flux(flux=F)` · `.add_flux(flux_x=F, flux_y=G)`

Adds the **conservative form** `−∂F(u)/∂x`. Use this for conservation laws where `F(u)` is
a nonlinear function of the field. The wave speed `a(u) = dF/du` is inferred automatically
via finite difference, and upwinding is based on `sign(a)` — **not** `sign(F)`.

This is the correct tool for Burgers, traffic flow, shallow water, and similar equations.

- **1D:** `−∂F(u)/∂x`
- **2D:** `−∂F(u)/∂x − ∂G(u)/∂y`

```python
# Burgers: F(u) = u²/2,  wave speed a = u
eq.add_flux(flux=lambda u: 0.5 * u**2)

# Traffic flow: F(u) = u(1−u),  wave speed a = 1−2u  (changes sign at u=0.5)
eq.add_flux(flux=lambda u: u * (1 - u))

# 2D Burgers
eq.add_flux(flux_x=lambda u: 0.5*u**2, flux_y=lambda u: 0.5*u**2)

# Central scheme (no wave speed needed)
eq.add_flux(flux=lambda u: 0.5*u**2, scheme='central')
```

Flux callables take the **field array directly**: `F(phi) -> ndarray`. Unlike other coefficient
callables they do not receive `x`, `y`, or other fields.

---

#### `.add_term(fn)`

Escape hatch for any term that doesn't fit the templates above. `fn` receives BC-aware
differential operator closures injected by argument name. This allows writing the RHS in
near-mathematical notation. See [Operator Reference](#operator-reference) for the full list.

```python
# Nonlinear scalar transport: -u*dT/dx + d(α(T)*dT/dx)/dx
def rhs(u, T, Dx, Div_flux_x, Div_flux_y):
    return -u * Dx(T, 'upwind', c=u) + Div_flux_x(alpha(T), T) + Div_flux_y(alpha(T), T)
eq_T.add_term(rhs)

# Pressure equation: dp/dt = -β(∂u/∂x + ∂v/∂y)
eq_p.add_term(lambda u, v, Dx, Dy: -beta * (Dx(u) + Dy(v)))
```

`add_term` may be combined freely with `add_advection`, `add_diffusion`, etc. — contributions
are summed.

---

#### `.set_bc(kind, side=None, mask=None, value=None)`

Sets a boundary condition. Multiple calls accumulate.

| `kind` | Meaning | `value` |
|--------|---------|---------|
| `'dirichlet'` | Prescribed field value | scalar or `callable(t)` |
| `'neumann'` | Prescribed outward normal derivative | scalar or `callable(t)` |
| `'periodic'` | Wrap-around | — |

```python
eq.set_bc(side='left',  kind='dirichlet', value=1.0)
eq.set_bc(side='right', kind='neumann',   value=0.0)   # insulating wall
eq.set_bc(kind='periodic')                             # 1D periodic
eq.set_bc(side='left',  kind='dirichlet', value=lambda t: np.sin(t))  # time-varying
```

**Side shortcuts:** `'left'`/`'right'` map to x-boundaries; `'bottom'`/`'top'` map to
y-boundaries (2D only). For explicit control pass `mask=bool_array`.

---

#### `.set_interior_bc(mask, kind='dirichlet', value=None)`

Marks an interior region as a solid obstacle or inclusion.

```python
cyl = (X - cx)**2 + (Y - cy)**2 <= r**2

eq.set_interior_bc(cyl, kind='dirichlet', value=2.0)      # heated cylinder
eq.set_interior_bc(cyl, kind='dirichlet', value=0.0)      # no-slip wall
eq.set_interior_bc(cyl, kind='neumann')                   # insulating obstacle (approx.)
eq.set_interior_bc(cyl, kind='dirichlet', value=lambda t: np.sin(t))  # oscillating
```

> **Note:** `kind='neumann'` freezes interior cells. This is a first-order approximation
> sufficient for pressure obstacles. It does not rigorously enforce zero normal flux at
> curved boundaries.

---

#### `.set_ic(ic)`

Sets the initial condition.

```python
eq.set_ic(0.0)                                    # uniform zero
eq.set_ic(lambda x: np.sin(x))                    # 1D callable
eq.set_ic(lambda x, y: np.sin(np.pi*x)*np.cos(y)) # 2D callable (meshgrid arrays)
eq.set_ic(np.load('field.npy'))                   # from array
```

---

### PDESystem

```python
PDESystem(equations)
```

Couples a list of PDE descriptors, validates at construction time, and solves.

#### `.solve(t_span, method='RK45', t_eval=None, **kwargs)`

| Parameter | Description |
|-----------|-------------|
| `t_span` | `(t0, tf)` |
| `method` | `'RK45'` (default), `'RK23'`, `'BDF'`, `'Radau'`, `'DOP853'` |
| `t_eval` | Times at which to store output. If `None`, integrator chooses steps. |
| `rtol`, `atol` | Tolerances forwarded to `solve_ivp`. |

---

### PDESolution

| Attribute | Shape | Description |
|-----------|-------|-------------|
| `sol.<field>` | `(nx, nt)` or `(nx, ny, nt)` | Solution array accessed by field name |
| `sol.t` | `(nt,)` | Time points |
| `sol.success` | `bool` | Integrator converged |
| `sol.message` | `str` | Integrator status |

---

## Operator Reference

uPDE exposes two levels of discretisation. Use `add_advection`/`add_diffusion`/`add_flux` for
standard terms. Drop into `add_term` with injected operators for anything non-standard.

### Level 1 — High-level methods

| Method | Term | Form | Upwind direction |
|--------|------|------|-----------------|
| `add_advection(velocity=c)` | $-c\,\partial\phi/\partial x$ | Convective | `sign(c)` |
| `add_advection(velocity_x=cx, velocity_y=cy)` | $-c_x\partial\phi/\partial x - c_y\partial\phi/\partial y$ | Convective | `sign(cx)`, `sign(cy)` |
| `add_diffusion(diffusivity=D)` | $\nabla\cdot(D\nabla\phi)$ | Conservative central | — |
| `add_diffusion(diffusivity_x=Dx, diffusivity_y=Dy)` | anisotropic version | Conservative central | — |
| `add_source(expr=S)` | $S(x[,y], \texttt{**fields})$ | Pointwise | — |
| `add_flux(flux=F)` | $-\partial F(\phi)/\partial x$ | Conservative | `sign(dF/dφ)` — inferred |
| `add_flux(flux_x=F, flux_y=G)` | $-\partial F/\partial x - \partial G/\partial y$ | Conservative | `sign(dF/dφ)` — inferred |
| `add_flux(..., scheme='central')` | same | Conservative central | — |

**`add_advection` vs `add_flux`:** `add_advection` is the convective form $c\,\partial\phi/\partial x$
and requires `c` to be the true characteristic speed. `add_flux` is the conservation form
$\partial F(\phi)/\partial x$ where the wave speed is $dF/d\phi$ — which may differ from $F/\phi$
when $F$ is nonlinear. For Burgers ($F = u^2/2$) the wave speed is $u$, not $u/2$.

### Level 2 — Injected operators for `add_term`

When `add_term(fn)` is called, uPDE inspects the argument names of `fn` and injects a
BC-aware operator closure for each recognised name. Field arrays from the coupled system
are also injected by name. All operators are **2nd-order central** unless noted.

**`phi`, `c`, `k`** each accept: ndarray · scalar · string field name.

**Hard rule: never nest operators.** Write `Dxx(phi)` not `Dx(Dx(phi))` — the intermediate
array carries no BC information and the result will be wrong at boundaries.

#### First derivatives

| Operator | Term | Scheme | Notes |
|----------|------|--------|-------|
| `Dx(phi)` | $\partial\phi/\partial x$ | 2nd-order central | 1D and 2D |
| `Dx(phi, 'upwind', c=v)` | $\partial\phi/\partial x$ | 1st-order upwind on `sign(v)` | 1D and 2D |
| `Dy(phi)` | $\partial\phi/\partial y$ | 2nd-order central | 2D only |
| `Dy(phi, 'upwind', c=v)` | $\partial\phi/\partial y$ | 1st-order upwind on `sign(v)` | 2D only |

```python
# Pressure gradient
eq.add_term(lambda p, Dx, Dy: -(1/rho) * Dx(p))

# Upwind convective advection: u*∂T/∂x + v*∂T/∂y
eq.add_term(lambda u, v, T, Dx, Dy:
    -u * Dx(T, 'upwind', c=u) - v * Dy(T, 'upwind', c=v))
```

#### Second derivatives

| Operator | Term | Scheme | Notes |
|----------|------|--------|-------|
| `Dxx(phi)` | $\partial^2\phi/\partial x^2$ | 2nd-order central, Dirichlet ghost | 1D and 2D |
| `Dyy(phi)` | $\partial^2\phi/\partial y^2$ | 2nd-order central, Dirichlet ghost | 2D only |

```python
# Laplacian with explicit viscosity
eq.add_term(lambda u, Dxx, Dyy: nu * (Dxx(u) + Dyy(u)))
```

#### Conservative flux divergence

| Operator | Term | Scheme | Notes |
|----------|------|--------|-------|
| `Div_x(c, phi)` | $-\partial(c\,\phi)/\partial x$ | 2nd-order central | 1D and 2D |
| `Div_y(c, phi)` | $-\partial(c\,\phi)/\partial y$ | 2nd-order central | 2D only |
| `Div_flux_x(k, phi)` | $\partial(k\,\partial\phi/\partial x)/\partial x$ | Conservative central | 1D and 2D |
| `Div_flux_y(k, phi)` | $\partial(k\,\partial\phi/\partial y)/\partial y$ | Conservative central | 2D only |

```python
# Continuity / divergence of velocity
eq_p.add_term(lambda u, v, Div_x, Div_y: Div_x(1.0, u) + Div_y(1.0, v))

# Variable diffusivity written explicitly
eq.add_term(lambda T, Div_flux_x, Div_flux_y:
    Div_flux_x(alpha(T), T) + Div_flux_y(alpha(T), T))
```

`Div_x`/`Div_y` are the central conservative divergence — no upwinding. Use
`add_flux` or `Dx(phi, 'upwind', c=v)` when you need upwinding.

### Coefficient conventions

All high-level method coefficients and `add_term` field arguments accept:

| Type | Example | Behaviour |
|------|---------|-----------|
| Scalar | `1.0` | Broadcast to grid |
| ndarray | `D_array` | Used directly |
| `callable(x, **fields)` | `lambda x, T: 1+T**2` | Called each timestep |
| `callable(x, y, **fields)` | `lambda x, y, T: ...` | 2D version |
| String field name | `'u'` | Looked up from coupled system each timestep |

---

## Examples

### Linear Advection

Gaussian pulse at speed `c=1` with periodic BCs — returns to start at `t = 2π`.

```python
x  = np.linspace(0, 2*np.pi, 256)
ic = np.exp(-((x - np.pi/3)**2) / 0.05)

eq = PDE('phi', x=x)
eq.add_advection(velocity=1.0)
eq.set_bc(kind='periodic')
eq.set_ic(ic)

sol = PDESystem([eq]).solve(t_span=(0, 6), method='RK45',
                             t_eval=np.linspace(0, 6, 120))
```

---

### Burgers Equation — Conservation Form

`F(u) = u²/2`, wave speed `a = u`. Use `add_flux` so upwinding is based on `sign(u)`, not `sign(F)`.

```python
x = np.linspace(0, 2*np.pi, 512, endpoint=False)

eq = PDE('u', x=x)
eq.add_flux(flux=lambda u: 0.5 * u**2)
eq.add_diffusion(diffusivity=0.005)
eq.set_bc(kind='periodic')
eq.set_ic(np.sin(x))

sol = PDESystem([eq]).solve(t_span=(0, 3), method='RK45',
                             t_eval=np.linspace(0, 3, 100))
```

---

### Traffic Flow — Nonlinear Conservation Law

`F(u) = u(1−u)`, wave speed `a = 1−2u` — changes sign at `u = 0.5`.
`add_flux` handles this correctly; `add_advection` would not.

```python
x = np.linspace(0, 1, 256, endpoint=False)

eq = PDE('rho', x=x)
eq.add_flux(flux=lambda rho: rho * (1 - rho))
eq.set_bc(kind='periodic')
eq.set_ic(0.3 + 0.2 * np.sin(2 * np.pi * x))

sol = PDESystem([eq]).solve(t_span=(0, 1), method='RK45')
```

---

### Reaction-Diffusion A + B → C

Three coupled species diffusing in from opposite ends and reacting where they meet.

```python
x  = np.linspace(0, 1, 256)
k1 = 10.0

eqA = PDE('cA', x=x)
eqA.add_diffusion(diffusivity=1.0)
eqA.add_source(expr=lambda x, cA, cB: -k1 * cA * cB)
eqA.set_bc(side='left',  kind='dirichlet', value=5.0)
eqA.set_bc(side='right', kind='dirichlet', value=0.0)
eqA.set_ic(0.0)

eqB = PDE('cB', x=x)
eqB.add_diffusion(diffusivity=1.0)
eqB.add_source(expr=lambda x, cA, cB: -k1 * cA * cB)
eqB.set_bc(side='left',  kind='dirichlet', value=0.0)
eqB.set_bc(side='right', kind='dirichlet', value=3.0)
eqB.set_ic(0.0)

eqC = PDE('cC', x=x)
eqC.add_diffusion(diffusivity=1.0)
eqC.add_source(expr=lambda x, cA, cB: +k1 * cA * cB)
eqC.set_bc(side='left',  kind='dirichlet', value=0.0)
eqC.set_bc(side='right', kind='dirichlet', value=0.0)
eqC.set_ic(0.0)

sol = PDESystem([eqA, eqB, eqC]).solve(t_span=(0, 0.5), method='RK45',
                                        rtol=1e-6, atol=1e-8)
# sol.cA, sol.cB, sol.cC — each shape (256, nt)
```

---

### 2D Heat Equation

Constant diffusivity on a unit square. Steady state: `T = 1 − x`.

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

sol = PDESystem([eq]).solve(t_span=(0, 1), method='BDF',
                             rtol=1e-4, atol=1e-6,
                             t_eval=np.linspace(0, 1, 10))
# sol.T.shape == (64, 64, 10)
# sol.T[:, :, -1]   final 2D field
# sol.T[:, 32, -1]  centreline profile
```

---

### 2D Nonlinear Diffusion

Field-dependent diffusivity `κ(T) = 1 + T²`.

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

sol = PDESystem([eq]).solve(t_span=(0, 0.5), method='BDF',
                             rtol=1e-4, atol=1e-6)
```

---

### Scalar Transport with Temperature-Dependent Diffusivity — `add_term`

Combines upwind convection and nonlinear diffusion in a single `add_term` callable.

```python
alpha = lambda T: 0.01 * (1 + T**2)

def rhs(u, v, T, Dx, Dy, Div_flux_x, Div_flux_y):
    adv  = u * Dx(T, 'upwind', c=u) + v * Dy(T, 'upwind', c=v)
    diff = Div_flux_x(alpha(T), T) + Div_flux_y(alpha(T), T)
    return -adv + diff

eq_T.add_term(rhs)
```

---

### 2D Heated Cylinder (Interior BC)

Cylinder held at `T = 2`, outer walls cold.

```python
nx, ny = 64, 64
x = np.linspace(0, 4, nx)
y = np.linspace(0, 4, ny)
X, Y = np.meshgrid(x, y, indexing='ij')
cyl = (X - 2.0)**2 + (Y - 2.0)**2 <= 0.5**2

eq = PDE('T', x=x, y=y)
eq.add_diffusion(diffusivity=0.1)
eq.set_bc(side='left',   kind='dirichlet', value=0.0)
eq.set_bc(side='right',  kind='dirichlet', value=0.0)
eq.set_bc(side='bottom', kind='neumann',   value=0.0)
eq.set_bc(side='top',    kind='neumann',   value=0.0)
eq.set_interior_bc(cyl, kind='dirichlet', value=2.0)
eq.set_ic(0.0)

sol = PDESystem([eq]).solve(t_span=(0, 5), method='BDF',
                             rtol=1e-4, atol=1e-6)
```

---

## Numerical Methods

### `add_advection` — convective upwind

`add_advection(velocity=c)` discretises `c · ∂φ/∂x` with a first-order upwind stencil,
choosing direction based on `sign(c)`:

```
∂φ/∂x ≈  (φ_i − φ_{i−1}) / Δx    if c ≥ 0   (backward)
          (φ_{i+1} − φ_i) / Δx    if c < 0   (forward)
```

`c` must be the true characteristic speed of the equation. For variable-coefficient
advection `c = c(x,t)`, the convective and conservative forms differ by `φ ∂c/∂x` and
are **not** equivalent unless `c` is spatially uniform.

### `add_flux` — conservation-law upwind

`add_flux(flux=F)` discretises `∂F(φ)/∂x` for a nonlinear conservation law. The wave
speed `a(φ) = dF/dφ` is inferred via forward finite difference, and upwinding is based
on `sign(a)` — not `sign(F)`. Donor-cell scheme:

```
F_{i+½} = F_i      if a_i ≥ 0
           F_{i+1}  if a_i < 0
dF/dx ≈ (F_{i+½} − F_{i-½}) / Δx
```

This correctly handles nonlinear fluxes where the wave speed changes sign, such as
traffic flow `F = ρ(1−ρ)` with `a = 1−2ρ`.

### `add_diffusion` — conservative central differences

```
[∂(D ∂φ/∂x)/∂x]_i ≈ (D_{i+½}(φ_{i+1}−φ_i) − D_{i−½}(φ_i−φ_{i−1})) / Δx²
D_{i+½} = ½(D_i + D_{i+1})
```

Face-averaged diffusivities ensure conservation and correctness for nonlinear `D(φ)`.

### `add_term` operators — all central

All injected operators (`Dx`, `Dy`, `Dxx`, `Dyy`, `Div_x`, `Div_y`, `Div_flux_x`,
`Div_flux_y`) use 2nd-order central differences. For upwinding within `add_term` use
`Dx(phi, 'upwind', c=v)` or `Dy(phi, 'upwind', c=v)` explicitly.

### Ghost-cell padding

`numpy.roll` wraps arrays cyclically — wrong for non-periodic boundaries. uPDE pads
arrays with a ghost layer before each stencil. Ghost values replicate the boundary
(zero-gradient) for non-periodic edges. Dirichlet ghost values are reflected as
`ghost = 2·g − φ_bnd` for second-derivative operators. Actual BC enforcement
(zeroing Dirichlet rows, correcting Neumann rows) runs after stencil evaluation.

---

## Choosing a Solver

| Problem | Recommended | Why |
|---------|-------------|-----|
| Pure advection | `RK45` | Explicit, non-stiff |
| Advection + mild diffusion | `RK45` | CFL limit manageable |
| Diffusion-dominated 1D | `RK45` or `BDF` | Depends on grid resolution |
| Diffusion-dominated 2D | `BDF` or `Radau` | 2D stiffness ∝ 1/Δx² |
| Stiff reaction-diffusion | `BDF` or `Radau` | Fast reaction timescales |
| Nonlinear conservation laws | `RK45` | Explicit fine for MOL |

> **2D blow-up?** Almost always stiffness. Switch `method='RK45'` → `method='BDF'`.
> The explicit stability limit for 2D diffusion scales as Δt ~ Δx², which becomes
> extremely restrictive on fine grids.

---

## Troubleshooting

**Solution blows up immediately in 2D**
Switch to `method='BDF'`. Explicit methods require Δt ~ Δx² for diffusion — impractical on fine grids.

**`ValueError: unknown field reference 'u'`**
A callable references a field name not declared in the system. Make sure all fields are passed to `PDESystem`.

**Upwinding gives diffusive results**
First-order upwind adds numerical diffusion ∝ Δx. Refine the grid or add a small physical diffusion term to regularise.

**`add_flux` gives wrong results for linear advection**
Use `add_advection` for linear advection `c · ∂u/∂x`. `add_flux` is for conservation laws `∂F(u)/∂x` and infers the wave speed numerically, which adds unnecessary overhead and rounding for the linear case.
