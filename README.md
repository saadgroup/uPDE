# uPDE

> A lightweight Python library for solving 1D and 2D PDEs using the method of lines.

uPDE lets you describe a PDE — its terms, coefficients, boundary conditions, and initial
data — and hands the spatial right-hand side directly to `scipy.integrate.solve_ivp`.
You get adaptive step control, stiffness detection, and a choice of explicit and implicit
integrators without writing a single line of time-stepping code.

**Features:**
- 1D and 2D problems on uniform Cartesian grids
- Coupled multi-field systems
- Scalar, array, and callable (nonlinear, field-dependent, or time-dependent) coefficients
- Dirichlet, Neumann, and periodic boundary conditions
- Interior boundary conditions for obstacles and inclusions
- Pre-built equation prototypes for common PDE families
- Pure NumPy / SciPy — no compilation, no external mesh libraries

---

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Core Concepts](#core-concepts)
- [User Guide](#user-guide)
  - [Coefficients and Callables](#coefficients-and-callables)
  - [Time-Dependent Callables](#time-dependent-callables)
  - [Boundary Conditions](#boundary-conditions)
  - [Initial Conditions](#initial-conditions)
  - [Coupled Systems](#coupled-systems)
  - [2D Problems](#2d-problems)
  - [Interior Boundary Conditions](#interior-boundary-conditions)
  - [Choosing a Solver](#choosing-a-solver)
- [Pre-built Equations](#pre-built-equations)
  - [HeatEquation](#heatequation)
  - [AdvectionDiffusion](#advectiondiffusion)
  - [Burgers](#burgers)
  - [ConservationLaw](#conservationlaw)
  - [ReactionDiffusion](#reactiondiffusion)
  - [WaveEquation](#waveequation)
  - [GrayScott](#grayscott)
  - [NavierStokes2D](#navierstokes2d)
- [API Reference](#api-reference)
  - [PDE](#pde)
  - [PDESystem](#pdesystem)
  - [PDESolution](#pdesolution)
  - [Operator Reference](#operator-reference)
- [Numerical Methods](#numerical-methods)
- [Troubleshooting](#troubleshooting)

---

## Installation

uPDE has no compiled dependencies. Drop `upde.py` and `equations.py` into your project,
or install from the repository:

```bash
# editable install from source
git clone https://github.com/tsaad-dev/upde.git
cd upde
pip install -e .

# or just copy the two files
cp upde.py equations.py your_project/
```

**Requirements:**

| Package | Version |
|---------|---------|
| numpy   | ≥ 1.22  |
| scipy   | ≥ 1.8   |
| python  | ≥ 3.9   |

---

## Quick Start

```python
import numpy as np
from upde import PDE, PDESystem

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

Or use a pre-built equation prototype and call `solve` directly on it:

```python
from upde import PDESystem, HeatEquation

eq = HeatEquation('T', x=x, diffusivity=0.05)
eq.set_bc(side='left',  kind='dirichlet', value=1.0)
eq.set_bc(side='right', kind='dirichlet', value=0.0)
eq.set_ic(0.0)

sol = eq.solve(t_span=(0, 1), method='BDF')   # shorthand for uncoupled equations
```

---

## Core Concepts

### PDE — equation descriptor

A `PDE` object holds the mathematical description of one equation: its field name,
spatial grid, terms (advection, diffusion, source, flux), boundary conditions, and
initial condition. It never solves anything on its own.

`PDE.solve()` is available as a convenience for equations that do not reference any
external field. For coupled equations, use `PDESystem` explicitly.

### PDESystem — the solver

`PDESystem` takes a list of `PDE` objects, validates that all field references are
consistent, assembles the joint state vector, and calls `solve_ivp`. Even single-equation
problems can use `PDESystem([eq]).solve(...)`.

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
| 1D | `f(x, t, **fields)` | coordinates + time + all fields |
| 2D | `f(x, y)` | coordinates only |
| 2D | `f(x, y, t)` | coordinates + time |
| 2D | `f(x, y, fieldA)` | coordinates + one field |
| 2D | `f(x, y, t, fieldA)` | coordinates + time + one field |
| 2D | `f(x, y, t, **fields)` | coordinates + time + all fields |

The ordering rule is: **coordinates → t → field names**. `t` is a reserved name
and cannot be used as a field name. Fixed parameters go in via closures:

```python
k1 = 10.0
eq.add_source(expr=lambda x, cA, cB: -k1 * cA * cB)  # k1 captured by closure
```

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
eq.set_bc(kind='periodic')          # 1D
eq.set_bc(kind='periodic', side='x')    # 2D — wrap left ↔ right
eq.set_bc(kind='periodic', side='y')    # 2D — wrap bottom ↔ top
eq.set_bc(kind='periodic', side='all')  # 2D — fully periodic
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

```python
eq.set_ic(0.0)                                           # scalar constant
eq.set_ic(np.zeros(n))                                   # 1D array
eq.set_ic(lambda x: np.sin(np.pi * x))                  # 1D callable
eq.set_ic(lambda x, y: np.sin(np.pi*x)*np.sin(np.pi*y)) # 2D callable

# override at solve time (useful for parameter sweeps)
sol = PDESystem([eq]).solve(t_span=(0, 1), ICs={'T': new_ic})
```

---

### Coupled Systems

Multi-field problems are built by creating one `PDE` per field and passing them all to
`PDESystem`. Fields reference each other by string name. `PDESystem` validates all
references at construction time.

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

Pass a second grid `y` to the `PDE` constructor. Everything else uses the same API.

```python
x = np.linspace(0, 1, 64)
y = np.linspace(0, 1, 64)

eq = PDE('T', x=x, y=y)

eq.add_diffusion(diffusivity=0.05)                      # isotropic
eq.add_diffusion(diffusivity_x=0.1, diffusivity_y=0.005) # anisotropic
eq.add_advection(velocity_x=1.0, velocity_y=0.5)        # 2D advection

# sol.T has shape (64, 64, nt)
```

> **Stiffness:** 2D diffusion is always stiff — explicit stability requires
> Δt ∼ Δx². Use `method='BDF'` for diffusion-dominated 2D problems.

---

### Time-Dependent Callables

Any coefficient — velocity, diffusivity, or source — can depend on the current
solver time `t` by declaring it as a parameter after the spatial coordinates.
No other changes to your code are required.

```python
import numpy as np
from upde import PDE

x = np.linspace(0, 1, 128)

eq = PDE('C', x=x)

# Velocity that ramps up linearly over the first 10 seconds
eq.add_advection(velocity=lambda x, t: min(t / 10.0, 1.0) * np.ones_like(x))

# Source that shuts off after 6 hours
eq.add_source(expr=lambda x, t, C: (1.0 if t < 6*3600 else 0.0) - 0.01 * C)

# Diffusivity from a time-interpolated dataset (e.g. ERA5 reanalysis)
import scipy.interpolate as interp
D_interp = interp.interp1d(time_points, D_values)
eq.add_diffusion(diffusivity=lambda x, t: D_interp(t) * np.ones_like(x))

eq.set_bc(kind='periodic')
eq.set_ic(ic)
sol = eq.solve(t_span=(0, 86400), method='RK45')
```

For 2D, `t` comes after both spatial coordinates:

```python
eq2d = PDE('C', x=x, y=y)

# Time-varying 2D wind field
eq2d.add_advection(
    velocity_x=lambda x, y, t: interp_U(t) * np.ones_like(x),
    velocity_y=lambda x, y, t: interp_V(t) * np.ones_like(x),
)
```

> **Note:** `t` is a reserved name. `PDE('t', ...)` raises a `ValueError`.
> Use `'T'`, `'tau'`, or any other name for a temperature-like field.

---

### Interior Boundary Conditions

`set_interior_bc` enforces a condition on an arbitrary boolean mask at every RHS
evaluation — useful for obstacles, cylinders, and inclusions.

```python
# Circular obstacle — no-slip
cyl = (X - cx)**2 + (Y - cy)**2 < radius**2
eq.set_interior_bc(cyl, kind='dirichlet', value=0.0)

# Insulating inclusion — zero flux (approximate)
eq.set_interior_bc(inclusion_mask, kind='neumann')
```

**Limitations:** the mask representation is first-order accurate (staircase surface).
The Neumann condition freezes the interior RHS to zero, which approximates
$\partial\phi/\partial n = 0$ but does not enforce it exactly on curved boundaries.

---

### Choosing a Solver

| Method | Type | Best for |
|--------|------|----------|
| `'RK45'` | Explicit | Advection-dominated, 1D diffusion, default |
| `'RK23'` | Explicit | Quick exploratory runs |
| `'BDF'` | Implicit | 2D diffusion, stiff reaction-diffusion |
| `'Radau'` | Implicit | Very stiff systems, higher accuracy than BDF |
| `'DOP853'` | Explicit | Smooth non-stiff problems, high accuracy |

```python
sol = PDESystem([eq]).solve(
    t_span  = (0, 1),
    method  = 'BDF',
    rtol    = 1e-4,
    atol    = 1e-6,
    t_eval  = np.linspace(0, 1, 50),
)
```

---

## Pre-built Equations

`equations.py` provides factory functions for common PDE families.
All factories return plain `PDE` objects (or `NamedPDESystem` for multi-field systems)
so the full API remains available after construction.

```python
from upde import (HeatEquation, AdvectionDiffusion, Burgers,
                  ConservationLaw, ReactionDiffusion,
                  WaveEquation, GrayScott, NavierStokes2D)
```

**1D vs 2D:** all single-field factories support both 1D and 2D — pass `y` to activate
2D mode. `Burgers` is 1D only. `WaveEquation`, `GrayScott`, and `NavierStokes2D`
require `y` (2D only).

**Direct solve:** single-field equations that reference no external fields can be
solved without constructing a `PDESystem`:

```python
sol = eq.solve(t_span=(0, 1), method='BDF')
```

If the equation references an external field, `solve()` raises a `ValueError` with a
clear message directing you to use `PDESystem`.

---

### HeatEquation

$$\frac{\partial T}{\partial t} = \nabla \cdot (D\,\nabla T)$$

```python
HeatEquation(field, x, y=None, diffusivity=1.0)
```

| Parameter | Description |
|-----------|-------------|
| `field` | Field name string |
| `x` | 1D coordinate array |
| `y` | 1D coordinate array (activates 2D mode) |
| `diffusivity` | $D$ — scalar, array, or callable; may depend on space or the field |

```python
# 1D rod: T=1 on left, T=0 on right
eq = HeatEquation('T', x=x, diffusivity=0.01)
eq.set_bc(side='left',  kind='dirichlet', value=1.0)
eq.set_bc(side='right', kind='dirichlet', value=0.0)
eq.set_ic(0.0)
sol = eq.solve(t_span=(0, 1), method='BDF')

# 2D with spatially varying diffusivity
eq2 = HeatEquation('T', x=x, y=y, diffusivity=lambda x, y: 1 + x)
```

---

### AdvectionDiffusion

$$\frac{\partial T}{\partial t} + c\,\frac{\partial T}{\partial x}
= \nabla \cdot (D\,\nabla T)$$

```python
AdvectionDiffusion(field, x, y=None,
                   velocity=None, velocity_x=None, velocity_y=None,
                   diffusivity=0.0,
                   diffusivity_x=None, diffusivity_y=None)
```

Advection uses the convective form with first-order upwinding on `sign(c)`.
Diffusion is conservative central.
All coefficients may be scalars, arrays, callables, or string field names
(for coupling to a velocity field from another equation).

```python
# 1D — scalar transport with constant velocity
eq = AdvectionDiffusion('T', x=x, velocity=1.0, diffusivity=0.005)
eq.set_bc(kind='periodic')
eq.set_ic(np.exp(-((x - 0.3)**2) / 0.01))
sol = eq.solve(t_span=(0, 1), method='RK45')

# 2D — passive scalar advected by a velocity field from another equation
eq_T = AdvectionDiffusion('T', x=x, y=y,
                           velocity_x='u', velocity_y='v',
                           diffusivity=0.001)
# eq_T.solve() would raise ValueError here — it references 'u' and 'v'
# Must use PDESystem([eq_u, eq_v, eq_T]).solve(...)
```

---

### Burgers

$$\frac{\partial u}{\partial t} + \frac{\partial}{\partial x}\!\left(\frac{u^2}{2}\right)
= \nu\,\frac{\partial^2 u}{\partial x^2}$$

```python
Burgers(field, x, viscosity=0.0)
```

Written in conservation form with flux $F(u) = u^2/2$.
Wave speed $a = u$ is inferred automatically; upwinding on `sign(a)`.
Set `viscosity=0` for the inviscid case (shock forms in finite time).
**1D only.**

```python
x = np.linspace(0, 2*np.pi, 512)

# Viscous — shock is regularised
eq = Burgers('u', x=x, viscosity=0.05)
eq.set_bc(kind='periodic')
eq.set_ic(np.sin(x))
sol = eq.solve(t_span=(0, 3), method='RK45')

# Inviscid — wave steepens to a shock
eq_inv = Burgers('u', x=x, viscosity=0.0)
eq_inv.set_bc(kind='periodic')
eq_inv.set_ic(np.sin(x))
sol_inv = eq_inv.solve(t_span=(0, 1), method='RK45')
```

---

### ConservationLaw

$$\frac{\partial u}{\partial t} + \frac{\partial F(u)}{\partial x} = 0$$

```python
ConservationLaw(field, x, flux, flux_y=None, scheme='upwind')
```

| Parameter | Description |
|-----------|-------------|
| `flux` | Callable $F(u)$ — 1D flux, or x-direction flux in 2D |
| `flux_y` | Callable $G(u)$ — y-direction flux (2D only) |
| `scheme` | `'upwind'` (default) or `'central'` |

The wave speed $a = dF/du$ is inferred automatically via finite difference
and used to select the upwind direction.

```python
# LWR traffic flow: F(ρ) = ρ(1−ρ), wave speed a = 1−2ρ
eq = ConservationLaw('rho', x=x, flux=lambda rho: rho * (1 - rho))
eq.set_bc(kind='periodic')
eq.set_ic(0.2 + 0.6 * np.exp(-((x - 0.3)**2) / 0.005))
sol = eq.solve(t_span=(0, 0.5), method='RK45')

# Buckley-Leverett (two-phase flow)
eq2 = ConservationLaw('s', x=x,
                       flux=lambda s: s**2 / (s**2 + (1 - s)**2))
```

---

### ReactionDiffusion

$$\frac{\partial u}{\partial t} = \nabla \cdot (D\,\nabla u) + R(u,\,x,\,\ldots)$$

```python
ReactionDiffusion(field, x, y=None, diffusivity=1.0, reaction=None)
```

The reaction callable `R` may reference other coupled fields by name,
enabling multi-species systems to be built by combining multiple instances.

```python
# Fisher-KPP travelling wave
r  = 1.0; D = 0.5
eq = ReactionDiffusion('u', x=x, diffusivity=D,
                        reaction=lambda x, u: r * u * (1 - u))
eq.set_bc(side='left',  kind='dirichlet', value=1.0)
eq.set_bc(side='right', kind='dirichlet', value=0.0)
eq.set_ic(lambda x: (x < 1.0).astype(float))
sol = eq.solve(t_span=(0, 20), method='BDF')
# Theoretical wave speed: c* = 2*sqrt(r*D)

# FitzHugh-Nagumo (two coupled species)
eps, beta, gamma = 0.1, 0.5, 1.0
eq_v = ReactionDiffusion('v', x=x, diffusivity=1.0,
                          reaction=lambda x, v, w: v - v**3/3 - w)
eq_w = ReactionDiffusion('w', x=x, diffusivity=0.0,
                          reaction=lambda x, v, w: eps*(v - beta*w + gamma))
sol = PDESystem([eq_v, eq_w]).solve(t_span=(0, 100), method='BDF')
```

---

### WaveEquation

$$\frac{\partial^2 u}{\partial t^2} = c^2\,\nabla^2 u$$

Split into a first-order system:

$$\frac{\partial u}{\partial t} = u_t \qquad
\frac{\partial u_t}{\partial t} = c^2\,\nabla^2 u$$

```python
WaveEquation(u_name, ut_name, x, y=None, speed=1.0)
```

Returns a `NamedPDESystem` with attributes named after the two fields.
BCs and ICs must be set on both fields.

```python
wave = WaveEquation('u', 'ut', x=x, speed=1.0)

wave.u.set_bc(side='left',  kind='dirichlet', value=0.0)
wave.u.set_bc(side='right', kind='dirichlet', value=0.0)
wave.ut.set_bc(side='left',  kind='dirichlet', value=0.0)
wave.ut.set_bc(side='right', kind='dirichlet', value=0.0)

wave.u.set_ic(lambda x: np.sin(np.pi * x))   # plucked string
wave.ut.set_ic(0.0)                            # released from rest

sol = wave.solve(t_span=(0, 2), method='RK45')
# Analytical solution: u(x,t) = cos(π c t) sin(π x)
```

---

### GrayScott

$$\frac{\partial u}{\partial t} = D_u\,\nabla^2 u - uv^2 + F(1-u)$$
$$\frac{\partial v}{\partial t} = D_v\,\nabla^2 v + uv^2 - (F+k)v$$

```python
GrayScott(u_name, v_name, x, y=None,
          Du=2e-5, Dv=1e-5, F=0.04, k=0.06)
```

Returns a `NamedPDESystem`. The autocatalytic $uv^2$ term drives Turing-type
pattern formation. Pattern type depends on $F$ and $k$:

| Pattern | $F$ | $k$ |
|---------|-----|-----|
| Spots | 0.035 | 0.065 |
| Stripes | 0.040 | 0.060 |
| Solitons | 0.025 | 0.055 |
| Chaos | 0.026 | 0.051 |

```python
gs = GrayScott('u', 'v', x=x, y=y, Du=2e-5, Dv=1e-5, F=0.035, k=0.065)
gs.u.set_bc(kind='periodic')
gs.v.set_bc(kind='periodic')

u0 = np.ones((nx, ny));   v0 = np.zeros((nx, ny))
cx, cy = nx//2, ny//2;    r = nx//10
u0[cx-r:cx+r, cy-r:cy+r] = 0.5
v0[cx-r:cx+r, cy-r:cy+r] = 0.25

gs.u.set_ic(u0);  gs.v.set_ic(v0)
sol = gs.solve(t_span=(0, 3000), method='BDF', rtol=1e-4, atol=1e-6)
# sol.u, sol.v — shape (nx, ny, nt)
```

> **Solver note:** Gray-Scott is stiff. Always use `method='BDF'` or `'Radau'`.
> Pattern formation requires long integration times (t ~ 1000–5000).

---

### NavierStokes2D

$$\frac{\partial u}{\partial t} + u\frac{\partial u}{\partial x} + v\frac{\partial u}{\partial y}
= -\frac{1}{\rho}\frac{\partial p}{\partial x} + \nu\nabla^2 u$$

$$\frac{\partial v}{\partial t} + u\frac{\partial v}{\partial x} + v\frac{\partial v}{\partial y}
= -\frac{1}{\rho}\frac{\partial p}{\partial y} + \nu\nabla^2 v$$

$$\frac{\partial p}{\partial t} = -\beta\!\left(\frac{\partial u}{\partial x}
+ \frac{\partial v}{\partial y}\right) + \varepsilon\,\nabla^2 p$$

```python
NavierStokes2D(u_name, v_name, p_name, x, y,
               nu=0.01, rho=1.0, beta=0.5,
               pressure_stabilisation=None)
```

| Parameter | Description |
|-----------|-------------|
| `nu` | Kinematic viscosity |
| `rho` | Density |
| `beta` | Artificial compressibility parameter. Larger → tighter divergence constraint but stiffer system |
| `pressure_stabilisation` | $\varepsilon$ for $\varepsilon\nabla^2 p$ term. Default `None` sets $\varepsilon = 0.5\Delta x^2$ automatically to suppress checkerboard oscillations. Set to `0.0` to disable |

Returns a `NamedPDESystem` with attributes named after the three fields.
BCs and ICs must be set on all three fields.

**Field names are user-supplied** — pass whatever names you want:

```python
ns = NavierStokes2D('u', 'v', 'p', x=x, y=y, nu=0.01)
# or
ns = NavierStokes2D('vx', 'vy', 'pressure', x=x, y=y, nu=0.01)
```

**Typical BC pattern:**

```python
ns = NavierStokes2D('u', 'v', 'p', x=x, y=y, nu=0.01, rho=1.0, beta=0.5)

# u: inflow left, outflow right, no-slip top/bottom
ns.u.set_bc(side='left',   kind='dirichlet', value=U_in)
ns.u.set_bc(side='right',  kind='neumann',   value=0.0)
ns.u.set_bc(side='bottom', kind='dirichlet', value=0.0)
ns.u.set_bc(side='top',    kind='dirichlet', value=0.0)

# v: no-slip / outflow
ns.v.set_bc(side='left',   kind='dirichlet', value=0.0)
ns.v.set_bc(side='right',  kind='neumann',   value=0.0)
ns.v.set_bc(side='bottom', kind='dirichlet', value=0.0)
ns.v.set_bc(side='top',    kind='dirichlet', value=0.0)

# p: Neumann everywhere
for side in ('left', 'right', 'bottom', 'top'):
    ns.p.set_bc(side=side, kind='neumann', value=0.0)

# ICs
ns.u.set_ic(U_in);  ns.v.set_ic(0.0);  ns.p.set_ic(0.0)

sol = ns.solve(t_span=(0, 60), method='RK45', rtol=1e-4, atol=1e-6)
```

**Cylinder obstacle:**

```python
cyl = (X - cx)**2 + (Y - cy)**2 < radius**2
ns.u.set_interior_bc(cyl, kind='dirichlet', value=0.0)
ns.v.set_interior_bc(cyl, kind='dirichlet', value=0.0)
ns.p.set_interior_bc(cyl, kind='neumann')
```

> **Important:** The artificial compressibility method is not divergence-free.
> The residual $\nabla\cdot\mathbf{u} = O(1/\beta)$ is expected behaviour, not a bug.
> For tighter incompressibility, increase $\beta$ (at the cost of stiffness).

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
| `add_flux(flux=, scheme=)` | 1D conservation law $-\partial_x F(\phi)$, wave speed inferred |
| `add_flux(flux_x=, flux_y=, scheme=)` | 2D conservation law |
| `add_term(fn)` | Generic operator term — see [Operator Reference](#operator-reference) |
| `set_bc(kind, side=, mask=, value=)` | Domain boundary condition |
| `set_interior_bc(mask, kind, value=)` | Interior obstacle BC |
| `set_ic(ic)` | Initial condition: scalar, array, or callable |
| `solve(t_span, **kwargs)` | Solve directly (uncoupled equations only) |

---

### PDESystem

```python
PDESystem(equations)
sol = PDESystem([eq1, eq2, ...]).solve(t_span, ICs=None, method='RK45',
                                       t_eval=None, **kwargs)
```

Validates all field references at construction time.
`**kwargs` are forwarded to `scipy.integrate.solve_ivp`.

---

### PDESolution

| Attribute | Description |
|-----------|-------------|
| `sol.<field>` | Field array — shape `(nx, nt)` in 1D, `(nx, ny, nt)` in 2D |
| `sol.t` | Time points `(nt,)` |
| `sol.success` | `True` if the integrator reached `t_span[1]` |
| `sol.message` | Integrator status string |
| `sol.raw` | Raw `scipy.OdeResult` |

---

### Operator Reference

Operators are injected into `add_term` functions by **name matching** — the order of
arguments does not matter. Any parameter whose name matches an operator or a declared
field is automatically injected; the rest are ignored.

```python
# All equivalent — order doesn't matter, only names do
def my_rhs(u, Dx, Dxx):       ...
def my_rhs(Dxx, Dx, u):       ...
def my_rhs(Dx, u, Dxx):       ...

# Use **fields to receive all coupled fields at once
def my_rhs(Dx, Dy, **fields):
    u = fields['u'];  v = fields['v']
    ...
```

#### Available operators

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

> **Rule:** never nest operators — use `Dxx(phi)` not `Dx(Dx(phi))`.

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
| Time integration | `solve_ivp` adaptive | depends on method |

All spatial operators are BC-aware — ghost cells are constructed from the boundary
conditions before each stencil application.

---

## Troubleshooting

**Solver fails or takes tiny steps**
- Switch to `method='BDF'` for diffusion-dominated or stiff problems
- Loosen tolerances: `rtol=1e-3, atol=1e-5`
- Check that BCs are consistent with the IC at `t=0`

**Solution blows up**
- Add or increase diffusivity to stabilise advection
- Reduce `beta` in NavierStokes2D if the pressure equation is causing instability
- Check that source terms don't have the wrong sign

**Checkerboard oscillations in pressure (NavierStokes2D)**
- The default `pressure_stabilisation=None` sets $\varepsilon = 0.5\Delta x^2$ automatically
- If oscillations persist, increase it: `pressure_stabilisation=dx**2`
- Do not set it so large that the physical pressure gradient is smoothed away

**`ValueError: references external field(s)`**
- `PDE.solve()` only works for uncoupled equations
- Use `PDESystem([eq1, eq2, ...]).solve(...)` for coupled problems

**Field not found in solution**
- Check the field name string passed to `PDE(field, ...)` matches what you access on `sol`
- Field names are case-sensitive: `'T'` and `'t'` are different
- `'t'` is reserved for the solver time — `PDE('t', ...)` raises `ValueError`
