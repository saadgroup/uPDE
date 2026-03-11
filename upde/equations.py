"""
pde_equations.py
================
Pre-packaged equation factories for uPDE.

Each factory returns one or more configured PDE objects.  The caller is
always responsible for setting boundary conditions, initial conditions,
and assembling the system:

    from pde_solver   import PDESystem
    from pde_equations import AdvectionDiffusion, NavierStokes2D

Single-field equations return a PDE:
    eq = AdvectionDiffusion('T', x=x, velocity=1.0, diffusivity=0.01)
    eq.set_bc(...)
    eq.set_ic(...)
    sol = PDESystem([eq]).solve(...)

Multi-field equations return a PDESystem with named equation attributes:
    ns = NavierStokes2D('u', 'v', 'p', x=x, y=y, nu=0.01)
    ns.u.set_bc(...)
    ns.v.set_bc(...)
    ns.p.set_bc(...)
    sol = ns.solve(...)

Available factories
-------------------
Single-field (return PDE):
    HeatEquation        -- du/dt = D * laplacian(u)
    AdvectionDiffusion  -- du/dt = -c*grad(u) + D*laplacian(u)
    Burgers             -- du/dt = -d(u^2/2)/dx + nu*d^2u/dx^2
    ConservationLaw     -- du/dt = -dF(u)/dx
    ReactionDiffusion   -- du/dt = D*laplacian(u) + R(u, ...)

Multi-field (return NamedPDESystem):
    WaveEquation        -- d^2u/dt^2 = c^2 * laplacian(u)  [fields: u, ut]
    GrayScott           -- two-species reaction-diffusion    [fields: u, v]
    NavierStokes2D      -- 2D NS via artificial compressibility [fields: u, v, p]
"""

import numpy as np
from upde import PDE, PDESystem


# ---------------------------------------------------------------------------
# NamedPDESystem — PDESystem with per-equation named attributes
# ---------------------------------------------------------------------------

class NamedPDESystem(PDESystem):
    """
    PDESystem subclass that exposes individual equations as named attributes.

    Constructed by the multi-field factory functions; not intended for
    direct instantiation by users.

    Example
    -------
    ns = NavierStokes2D('u', 'v', 'p', x=x, y=y, nu=0.01)
    ns.u.set_bc(side='left', kind='dirichlet', value=1.0)
    ns.v.set_bc(side='left', kind='dirichlet', value=0.0)
    sol = ns.solve(t_span=(0, 10), method='RK45')
    sol.u   # shape (nx, ny, nt)
    """

    def __init__(self, equations, names):
        """
        Parameters
        ----------
        equations : list[PDE]
        names     : list[str]
            Attribute names to attach, one per equation.
            names[i] becomes self.<names[i]> = equations[i].
        """
        super().__init__(equations)
        for name, eq in zip(names, equations):
            setattr(self, name, eq)

    def __repr__(self):
        fields = ', '.join(eq.field for eq in self.equations)
        return f"NamedPDESystem(fields=[{fields}])"


# ---------------------------------------------------------------------------
# 1. HeatEquation
# ---------------------------------------------------------------------------

def HeatEquation(field, x, y=None, diffusivity=1.0):
    """
    Heat equation (pure diffusion):

        du/dt = ∂(D ∂u/∂x)/∂x  [+ ∂(D ∂u/∂y)/∂y in 2D]

    Conservative central differences with face-averaged diffusivity.
    D may be spatially varying or field-dependent.

    Parameters
    ----------
    field       : str    -- field name, e.g. 'T'
    x           : array  -- 1D x-coordinate array
    y           : array or None -- 1D y-coordinate array (activates 2D mode)
    diffusivity : scalar | ndarray | callable(x[,y], **fields)

    Returns
    -------
    PDE

    Example
    -------
    eq = HeatEquation('T', x=x, diffusivity=0.01)
    eq.set_bc(side='left',  kind='dirichlet', value=1.0)
    eq.set_bc(side='right', kind='dirichlet', value=0.0)
    eq.set_ic(0.0)
    sol = PDESystem([eq]).solve(t_span=(0, 1), method='BDF')
    """
    eq = PDE(field, x=x, y=y)
    eq.add_diffusion(diffusivity=diffusivity)
    return eq


# ---------------------------------------------------------------------------
# 2. AdvectionDiffusion
# ---------------------------------------------------------------------------

def AdvectionDiffusion(field, x, y=None,
                       velocity=None,
                       velocity_x=None, velocity_y=None,
                       diffusivity=0.0,
                       diffusivity_x=None, diffusivity_y=None):
    """
    Advection-diffusion equation:

        du/dt = -c * ∂u/∂x + ∂(D ∂u/∂x)/∂x   [1D]

        du/dt = -cx*∂u/∂x - cy*∂u/∂y
                + ∂(Dx ∂u/∂x)/∂x + ∂(Dy ∂u/∂y)/∂y   [2D]

    Advection: convective form, first-order upwind on sign(c).
    Diffusion: conservative central, face-averaged D.

    Parameters
    ----------
    field           : str
    x               : array
    y               : array or None
    velocity        : scalar | ndarray | callable | str  [1D]
    velocity_x/y    : scalar | ndarray | callable | str  [2D]
    diffusivity     : scalar | ndarray | callable        [isotropic]
    diffusivity_x/y : scalar | ndarray | callable        [anisotropic 2D]

    Returns
    -------
    PDE

    Example — 1D
    ------------
    eq = AdvectionDiffusion('T', x=x, velocity=1.0, diffusivity=0.01)
    eq.set_bc(kind='periodic')
    eq.set_ic(np.exp(-((x - np.pi)**2) / 0.1))
    sol = PDESystem([eq]).solve(t_span=(0, 2), method='RK45')

    Example — 2D with field-coupled velocity
    ----------------------------------------
    eq_T = AdvectionDiffusion('T', x=x, y=y,
                               velocity_x='u', velocity_y='v',
                               diffusivity=0.001)
    """
    eq = PDE(field, x=x, y=y)

    # advection
    if y is not None:
        if velocity_x is not None or velocity_y is not None:
            eq.add_advection(velocity_x=velocity_x, velocity_y=velocity_y)
    else:
        if velocity is not None:
            eq.add_advection(velocity=velocity)

    # diffusion
    if diffusivity_x is not None or diffusivity_y is not None:
        eq.add_diffusion(diffusivity_x=diffusivity_x,
                         diffusivity_y=diffusivity_y)
    elif diffusivity is not None and np.any(np.asarray(diffusivity) != 0):
        eq.add_diffusion(diffusivity=diffusivity)

    return eq


# ---------------------------------------------------------------------------
# 3. Burgers
# ---------------------------------------------------------------------------

def Burgers(field, x, viscosity=0.0):
    """
    Viscous (or inviscid) Burgers equation:

        du/dt + d(u²/2)/dx = nu * d²u/dx²

    Conservation form for the flux F(u) = u²/2. Wave speed a = dF/du = u
    is inferred automatically; upwinding based on sign(u).
    Viscosity term uses 2nd-order central differences.

    1D only (the canonical Burgers equation is inherently 1D).

    Parameters
    ----------
    field     : str    -- field name, e.g. 'u'
    x         : array  -- 1D coordinate array
    viscosity : float  -- kinematic viscosity nu (0 = inviscid)

    Returns
    -------
    PDE

    Example — near-shock formation
    -------------------------------
    eq = Burgers('u', x=x, viscosity=0.005)
    eq.set_bc(kind='periodic')
    eq.set_ic(np.sin(x))
    sol = PDESystem([eq]).solve(t_span=(0, 3), method='RK45')

    Notes
    -----
    For the inviscid case (viscosity=0) the solution develops a true
    discontinuity in finite time. The first-order upwind scheme adds
    numerical diffusion that regularises this, but the shock will still
    sharpen considerably. Reduce the grid spacing to resolve it better.
    """
    eq = PDE(field, x=x)
    eq.add_flux(flux=lambda u: 0.5 * u**2)
    if viscosity != 0.0:
        eq.add_diffusion(diffusivity=viscosity)
    return eq


# ---------------------------------------------------------------------------
# 4. ConservationLaw
# ---------------------------------------------------------------------------

def ConservationLaw(field, x, flux, flux_y=None, scheme='upwind'):
    """
    Generic scalar conservation law:

        du/dt + dF(u)/dx = 0           [1D]
        du/dt + dF(u)/dx + dG(u)/dy = 0   [2D]

    The wave speed a(u) = dF/du is inferred automatically via finite
    difference; upwinding is based on sign(a), not sign(F).

    Parameters
    ----------
    field   : str
    x       : array
    flux    : callable(u) -> ndarray   [1D flux F, or 2D x-direction flux]
    flux_y  : callable(u) -> ndarray   [2D y-direction flux G, optional]
    scheme  : 'upwind' (default) | 'central'

    Returns
    -------
    PDE

    Examples
    --------
    # Traffic flow: F(rho) = rho*(1-rho), wave speed a = 1-2*rho
    eq = ConservationLaw('rho', x=x, flux=lambda rho: rho*(1-rho))
    eq.set_bc(kind='periodic')
    eq.set_ic(0.3 + 0.2*np.sin(2*np.pi*x))

    # Buckley-Leverett: F(s) = s^2 / (s^2 + (1-s)^2)
    eq = ConservationLaw('s', x=x,
                          flux=lambda s: s**2 / (s**2 + (1-s)**2))
    """
    eq = PDE(field, x=x)
    if flux_y is not None:
        eq.add_flux(flux_x=flux, flux_y=flux_y, scheme=scheme)
    else:
        eq.add_flux(flux=flux, scheme=scheme)
    return eq


# ---------------------------------------------------------------------------
# 5. ReactionDiffusion
# ---------------------------------------------------------------------------

def ReactionDiffusion(field, x, y=None, diffusivity=1.0, reaction=None):
    """
    Single-species reaction-diffusion equation:

        du/dt = D * laplacian(u) + R(u, x[,y], **fields)

    The reaction term R may reference other coupled fields, enabling
    multi-species systems to be built by combining multiple instances.

    Parameters
    ----------
    field       : str
    x           : array
    y           : array or None
    diffusivity : scalar | ndarray | callable
    reaction    : callable(x[,y], **fields) -> ndarray, or None

    Returns
    -------
    PDE

    Examples
    --------
    # Fisher-KPP travelling wave: du/dt = D*uxx + r*u*(1-u)
    r  = 1.0
    eq = ReactionDiffusion('u', x=x, diffusivity=0.01,
                            reaction=lambda x, u: r*u*(1-u))
    eq.set_bc(side='left',  kind='dirichlet', value=1.0)
    eq.set_bc(side='right', kind='dirichlet', value=0.0)
    eq.set_ic(lambda x: (x < 0.1).astype(float))

    # FitzHugh-Nagumo (two coupled species):
    eps, beta, gamma = 0.1, 0.5, 1.0
    eq_v = ReactionDiffusion('v', x=x, diffusivity=1.0,
                              reaction=lambda x, v, w: v - v**3/3 - w)
    eq_w = ReactionDiffusion('w', x=x, diffusivity=0.0,
                              reaction=lambda x, v, w: eps*(v - beta*w + gamma))
    sol  = PDESystem([eq_v, eq_w]).solve(...)
    """
    eq = PDE(field, x=x, y=y)
    if diffusivity is not None and np.any(np.asarray(diffusivity) != 0):
        eq.add_diffusion(diffusivity=diffusivity)
    if reaction is not None:
        eq.add_source(expr=reaction)
    return eq


# ---------------------------------------------------------------------------
# 6. WaveEquation
# ---------------------------------------------------------------------------

def WaveEquation(u_name, ut_name, x, y=None, speed=1.0):
    """
    Second-order wave equation split into a first-order system:

        ∂u/∂t  = ut
        ∂ut/∂t = c² * laplacian(u)

    where c is the wave speed.

    Parameters
    ----------
    u_name  : str   -- field name for displacement, e.g. 'u'
    ut_name : str   -- field name for velocity,     e.g. 'ut'
    x       : array
    y       : array or None
    speed   : scalar | ndarray | callable  -- wave speed c

    Returns
    -------
    NamedPDESystem  with attributes .u and .ut  (or whatever names you pass)

    Example
    -------
    wave = WaveEquation('u', 'ut', x=x, speed=1.0)
    wave.u.set_bc(side='left',  kind='dirichlet', value=0.0)
    wave.u.set_bc(side='right', kind='dirichlet', value=0.0)
    wave.ut.set_bc(side='left',  kind='dirichlet', value=0.0)
    wave.ut.set_bc(side='right', kind='dirichlet', value=0.0)
    wave.u.set_ic(lambda x: np.sin(np.pi * x))
    wave.ut.set_ic(0.0)
    sol = wave.solve(t_span=(0, 2), method='RK45')
    sol.u   # shape (nx, nt)

    Notes
    -----
    BCs must be set on both u and ut.  Typically u gets the physical
    boundary condition (Dirichlet or Neumann) and ut gets a matching
    homogeneous BC.

    The wave equation is non-dissipative: energy is conserved exactly
    by the continuous equation. The central-difference spatial scheme
    and RK45 time integration introduce small dispersive errors that
    grow over long integration times. For long simulations use a
    symplectic integrator or add a small amount of numerical damping.
    """
    c2 = speed**2 if np.isscalar(speed) else speed**2

    # u_t = ut  (pure source — use closure to capture ut_name)
    def make_u_source(ut_name):
        return lambda *args, **fields: fields[ut_name]

    eq_u = PDE(u_name, x=x, y=y)
    eq_u.add_source(expr=make_u_source(ut_name))

    # ut_t = c² * laplacian(u)  (use closure to capture u_name and c2)
    def make_ut_rhs(u_name, c2, has_y):
        if has_y:
            def ut_rhs(Dxx, Dyy, **fields):
                return c2 * (Dxx(fields[u_name]) + Dyy(fields[u_name]))
        else:
            def ut_rhs(Dxx, **fields):
                return c2 * Dxx(fields[u_name])
        return ut_rhs

    eq_ut = PDE(ut_name, x=x, y=y)
    eq_ut.add_term(make_ut_rhs(u_name, c2, y is not None))

    return NamedPDESystem([eq_u, eq_ut], [u_name, ut_name])


# ---------------------------------------------------------------------------
# 7. GrayScott
# ---------------------------------------------------------------------------

def GrayScott(u_name, v_name, x, y=None,
              Du=2e-5, Dv=1e-5, F=0.04, k=0.06):
    """
    Gray-Scott two-species reaction-diffusion system:

        ∂u/∂t = Du * laplacian(u) - u*v² + F*(1 - u)
        ∂v/∂t = Dv * laplacian(v) + u*v² - (F + k)*v

    This is the canonical model for Turing-type pattern formation.
    Depending on F and k, the system produces spots, stripes, solitons,
    or spatiotemporal chaos.

    Parameters
    ----------
    u_name : str   -- activator field name, e.g. 'u'
    v_name : str   -- inhibitor field name, e.g. 'v'
    x      : array
    y      : array or None
    Du     : float -- diffusivity of u  (default 2e-5)
    Dv     : float -- diffusivity of v  (default 1e-5)
    F      : float -- feed rate         (default 0.04)
    k      : float -- kill rate         (default 0.06)

    Returns
    -------
    NamedPDESystem  with attributes .u and .v

    Example
    -------
    gs = GrayScott('u', 'v', x=x, y=y, Du=2e-5, Dv=1e-5, F=0.04, k=0.06)
    gs.u.set_bc(kind='periodic')
    gs.v.set_bc(kind='periodic')
    # IC: uniform steady state with small random perturbation in centre
    u0 = np.ones((nx, ny))
    v0 = np.zeros((nx, ny))
    cx, cy = nx//2, ny//2; r = nx//10
    u0[cx-r:cx+r, cy-r:cy+r] = 0.5
    v0[cx-r:cx+r, cy-r:cy+r] = 0.25
    u0 += 0.01 * np.random.randn(nx, ny)
    v0 += 0.01 * np.random.randn(nx, ny)
    gs.u.set_ic(u0)
    gs.v.set_ic(v0)
    sol = gs.solve(t_span=(0, 5000), method='BDF', rtol=1e-4, atol=1e-6)

    Notes
    -----
    Classic parameter sets:
        Spots:   F=0.035, k=0.065
        Stripes: F=0.040, k=0.060
        Solitons:F=0.025, k=0.055
        Chaos:   F=0.026, k=0.051

    Gray-Scott is stiff (fast reaction timescales): use method='BDF'.
    The domain should be large relative to the pattern wavelength
    (~1/sqrt(Du)) — typically L ~ 1 with nx=256 for 2D.
    """
    eq_u = PDE(u_name, x=x, y=y)
    eq_u.add_diffusion(diffusivity=Du)
    eq_u.add_source(
        expr=lambda *args, **fields: (
            -fields[u_name] * fields[v_name]**2
            + F * (1.0 - fields[u_name])
        )
    )

    eq_v = PDE(v_name, x=x, y=y)
    eq_v.add_diffusion(diffusivity=Dv)
    eq_v.add_source(
        expr=lambda *args, **fields: (
            fields[u_name] * fields[v_name]**2
            - (F + k) * fields[v_name]
        )
    )

    return NamedPDESystem([eq_u, eq_v], [u_name, v_name])


# ---------------------------------------------------------------------------
# 8. NavierStokes2D
# ---------------------------------------------------------------------------

def NavierStokes2D(u_name, v_name, p_name, x, y,
                   nu=0.01, rho=1.0, beta=0.5):
    """
    2D incompressible Navier-Stokes via artificial compressibility:

        ∂u/∂t = -u∂u/∂x - v∂u/∂y - (1/ρ)∂p/∂x + ν∇²u
        ∂v/∂t = -u∂v/∂x - v∂v/∂y - (1/ρ)∂p/∂y + ν∇²v
        ∂p/∂t = -β(∂u/∂x + ∂v/∂y)

    The pressure evolution equation drives divergence toward zero.
    The residual divergence is O(1/β) — this method is not exactly
    divergence-free. Increase β for tighter incompressibility at the
    cost of increased stiffness.

    Advection terms use the convective form with first-order upwinding.
    Viscous terms and pressure gradients use 2nd-order central differences.

    Parameters
    ----------
    u_name : str   -- x-velocity field name, e.g. 'u'
    v_name : str   -- y-velocity field name, e.g. 'v'
    p_name : str   -- pressure field name,   e.g. 'p'
    x      : array -- 1D x-coordinate array
    y      : array -- 1D y-coordinate array
    nu     : float -- kinematic viscosity  (default 0.01)
    rho    : float -- density              (default 1.0)
    beta   : float -- artificial compressibility parameter (default 0.5)

    Returns
    -------
    NamedPDESystem  with attributes .u, .v, .p

    Example — flow over a cylinder
    --------------------------------
    ns = NavierStokes2D('u', 'v', 'p', x=x, y=y, nu=0.01, rho=1.0, beta=0.5)

    # Boundary conditions
    ns.u.set_bc(side='left',   kind='dirichlet', value=U_in)
    ns.u.set_bc(side='right',  kind='neumann',   value=0.0)
    ns.u.set_bc(side='bottom', kind='dirichlet', value=0.0)
    ns.u.set_bc(side='top',    kind='dirichlet', value=0.0)

    ns.v.set_bc(side='left',   kind='dirichlet', value=0.0)
    ns.v.set_bc(side='right',  kind='neumann',   value=0.0)
    ns.v.set_bc(side='bottom', kind='dirichlet', value=0.0)
    ns.v.set_bc(side='top',    kind='dirichlet', value=0.0)

    for side in ('left', 'right', 'bottom', 'top'):
        ns.p.set_bc(side=side, kind='neumann', value=0.0)

    # Cylinder obstacle
    cyl = (X - cx)**2 + (Y - cy)**2 < radius**2
    ns.u.set_interior_bc(cyl, kind='dirichlet', value=0.0)
    ns.v.set_interior_bc(cyl, kind='dirichlet', value=0.0)
    ns.p.set_interior_bc(cyl, kind='neumann')

    # Initial conditions
    u0 = np.full((Nx, Ny), U_in); u0[cyl] = 0.0
    ns.u.set_ic(u0)
    ns.v.set_ic(0.0)
    ns.p.set_ic(0.0)

    sol = ns.solve(t_span=(0, 60), method='RK45', rtol=1e-4, atol=1e-6)
    sol.u   # shape (Nx, Ny, nt)

    Notes
    -----
    Recommended solver: method='RK45' for advection-dominated flows
    (moderate Re). Switch to method='BDF' for low-Re viscous flows or
    fine grids where diffusion stiffness dominates.

    The artificial compressibility method is not divergence-free.
    The residual ∇·u ~ O(1/β) is physical — it is the price paid for
    avoiding a pressure-Poisson solve. It does not indicate a bug.
    """
    def u_rhs(**fields):
        raise NotImplementedError  # replaced below

    # Build RHS closures capturing the field names and parameters
    def make_u_rhs(u_name, v_name, p_name, nu, rho):
        def u_rhs(Dx, Dy, Dxx, Dyy, **fields):
            u = fields[u_name]; v = fields[v_name]; p = fields[p_name]
            adv = u * Dx(u, 'upwind', c=u) + v * Dy(u, 'upwind', c=v)
            return -adv - (1/rho) * Dx(p) + nu * (Dxx(u) + Dyy(u))
        return u_rhs

    def make_v_rhs(u_name, v_name, p_name, nu, rho):
        def v_rhs(Dx, Dy, Dxx, Dyy, **fields):
            u = fields[u_name]; v = fields[v_name]; p = fields[p_name]
            adv = u * Dx(v, 'upwind', c=u) + v * Dy(v, 'upwind', c=v)
            return -adv - (1/rho) * Dy(p) + nu * (Dxx(v) + Dyy(v))
        return v_rhs

    def make_p_rhs(u_name, v_name, beta):
        def p_rhs(Dx, Dy, **fields):
            u = fields[u_name]; v = fields[v_name]
            return -beta * (Dx(u) + Dy(v))
        return p_rhs

    eq_u = PDE(u_name, x=x, y=y)
    eq_u.add_term(make_u_rhs(u_name, v_name, p_name, nu, rho))

    eq_v = PDE(v_name, x=x, y=y)
    eq_v.add_term(make_v_rhs(u_name, v_name, p_name, nu, rho))

    eq_p = PDE(p_name, x=x, y=y)
    eq_p.add_term(make_p_rhs(u_name, v_name, beta))

    return NamedPDESystem([eq_u, eq_v, eq_p], [u_name, v_name, p_name])
