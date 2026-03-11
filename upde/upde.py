"""
pde_solver.py
=============
A 1D and 2D PDE solver built on top of scipy's solve_ivp (method of lines).

Design
------
  PDE        — descriptor for a single field equation. Never solves anything.
  PDESystem  — owns a list of PDE objects, validates coupling, and solves.

Dimensionality
--------------
  1D : PDE('T', x=x)           — x is a 1D array of shape (nx,)
  2D : PDE('T', x=x, y=y)      — x, y are 1D arrays; meshgrid built internally

  Fields are:
    1D : numpy array of shape (nx,)
    2D : numpy array of shape (nx, ny)

  PDESolution fields:
    1D : (nx, nt)
    2D : (nx, ny, nt)   — spatial indices first, time last

Coefficient / expr callables
-----------------------------
  1D : f(x, **fields)           — x is 1D array (nx,)
  2D : f(x, y, **fields)        — x, y are 2D meshgrid arrays (nx, ny)
  Extra parameters captured via closures.
  Scalars and ndarrays are also accepted and broadcast appropriately.

Example — 2D heat equation
---------------------------
    import numpy as np
    from pde_solver import PDE, PDESystem

    x = np.linspace(0, 1, 64)
    y = np.linspace(0, 1, 64)

    eq = PDE('T', x=x, y=y)
    eq.add_diffusion(diffusivity=0.01)
    eq.set_bc(side='left',   kind='dirichlet', value=1.0)
    eq.set_bc(side='right',  kind='dirichlet', value=0.0)
    eq.set_bc(side='bottom', kind='neumann',   value=0.0)
    eq.set_bc(side='top',    kind='neumann',   value=0.0)
    eq.set_ic(lambda x, y: np.zeros_like(x))

    sol = PDESystem([eq]).solve(t_span=(0, 1), method='RK45')
    # sol.T  ->  shape (64, 64, nt)
    # sol.T[:, :, -1]  ->  final 2D field

Example — 1D coupled reaction-diffusion
-----------------------------------------
    x  = np.linspace(0, 1, 256)
    k1 = 10.0

    eqA = PDE('cA', x=x)
    eqA.add_diffusion(diffusivity=1.0)
    eqA.add_source(expr=lambda x, cA, cB, cC: -k1 * cA * cB)
    eqA.set_bc(side='left',  kind='dirichlet', value=5.0)
    eqA.set_bc(side='right', kind='dirichlet', value=0.0)
    eqA.set_ic(0.0)

    # ... eqB, eqC similarly ...

    sol = PDESystem([eqA, eqB, eqC]).solve(t_span=(0, 0.5), method='RK45')
    # sol.cA  ->  shape (256, nt)
"""

import inspect
import numpy as np
from scipy.integrate import solve_ivp


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_callable(val):
    """
    Normalise a coefficient/expr to a callable f(x, **fields) (1D)
    or f(x, y, **fields) (2D).  The callable convention is detected
    at call time by _call(); here we just wrap scalars and arrays.

    String values are treated as field name references: the returned
    callable looks the name up in **fields at call time, so passing
    e.g. velocity_x='u' works for coupled-field advection.
    """
    if callable(val):
        return val
    if isinstance(val, str):
        name = val
        def _field_ref(x, *args, **kw):
            if name not in kw:
                raise KeyError(
                    f"Field reference '{name}' not found in system fields. "
                    f"Available: {list(kw.keys())}"
                )
            return kw[name]
        _field_ref._field_ref_name = name   # tag for validation
        return _field_ref
    if np.isscalar(val):
        def _const(x, *args, **kw):
            return np.full_like(x, float(val))
        return _const
    if isinstance(val, np.ndarray):
        arr = val.copy()
        def _arr(x, *args, **kw):
            return arr
        return _arr
    raise TypeError(
        f"Coefficient must be scalar, ndarray, callable, or field-name string; got {type(val)}"
    )


def _call(fn, coords, fields):
    """
    Call fn with spatial coordinates and field arrays.

    coords : tuple — (x,) in 1D, (x, y) in 2D  (meshgrid arrays)
    fields : dict  — {field_name: array, ...}

    Forwards only the kwargs the callable declares; **kwargs gets everything.
    'x' and 'y' are positional, fields are keyword.
    """
    sig    = inspect.signature(fn)
    params = sig.parameters
    has_var_kw = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
    )
    if has_var_kw:
        return fn(*coords, **fields)
    accepted = {k: v for k, v in fields.items() if k in params}
    return fn(*coords, **accepted)


def _callable_field_refs(fn):
    """
    Return parameter names of fn that look like field references
    (not 'x', 'y', 't', operator names, and not *args / **kwargs).
    """
    if not callable(fn):
        return set()
    sig    = inspect.signature(fn)
    ignore = {'x', 'y', 't', 'Dx', 'Dy', 'Dxx', 'Dyy', 'Div_x', 'Div_y', 'Div_flux_x', 'Div_flux_y'}
    return {
        name
        for name, p in sig.parameters.items()
        if name not in ignore
        and p.kind not in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        )
    }


# ---------------------------------------------------------------------------
# Differential operators — BC-aware, injected into add_term callables
# ---------------------------------------------------------------------------

def _build_operators(eq, fields_nd, bct):
    """
    Build BC-aware differential operators for use inside add_term callables.

    Returns a dict with keys:
      Dx, Dy       — first derivative, central differences
      Dxx, Dyy     — second derivative, central differences
                     Use these instead of Dx(Dx(phi)) — nesting Dx is wrong
                     because the intermediate array has no BC information.
      Div_x, Div_y — central flux divergence -d(c*phi)/d{x,y}

    phi / c may each be:
      - ndarray of shape (nx,) or (nx,ny)
      - a string field name  (looked up in fields_nd at eval time)
      - a scalar             (broadcast to grid shape)
    """
    def _resolve(phi):
        if isinstance(phi, str):
            return fields_nd[phi]
        if np.isscalar(phi):
            shape = (eq.nx, eq.ny) if eq.is_2d else (eq.nx,)
            return np.full(shape, float(phi))
        return phi

    def _bc_val(normal, t=0.0):
        """Get prescribed BC value for a given normal direction, or 0."""
        for bc in eq._bcs:
            if bc.normal == normal and bc.kind == 'dirichlet':
                return bc.get_value(t)
        return 0.0

    if eq.is_2d:
        dx, dy = eq.dx, eq.dy

        def Dx(phi, scheme='central', c=None):
            """∂phi/∂x. scheme='central' (default) or 'upwind' (requires c)."""
            phi = _resolve(phi)
            n   = phi.shape[0]
            phi_ext = _ghost_pad(phi, axis=0, bc_lo=bct['x-'], bc_hi=bct['x+'])
            phi_c = np.take(phi_ext, range(1, n+1), axis=0)
            phi_r = np.take(phi_ext, range(2, n+2), axis=0)
            phi_l = np.take(phi_ext, range(0, n),   axis=0)
            if scheme == 'central':
                return (phi_r - phi_l) / (2.0 * dx)
            elif scheme == 'upwind':
                if c is None:
                    raise ValueError("Dx(..., scheme='upwind') requires c=velocity")
                c = _resolve(c)
                return np.where(c >= 0, (phi_c - phi_l) / dx,
                                        (phi_r - phi_c) / dx)
            else:
                raise ValueError(f"scheme must be 'central' or 'upwind'; got '{scheme}'")

        def Dy(phi, scheme='central', c=None):
            """∂phi/∂y. scheme='central' (default) or 'upwind' (requires c)."""
            phi = _resolve(phi)
            n   = phi.shape[1]
            phi_ext = _ghost_pad(phi, axis=1, bc_lo=bct['y-'], bc_hi=bct['y+'])
            phi_c = np.take(phi_ext, range(1, n+1), axis=1)
            phi_r = np.take(phi_ext, range(2, n+2), axis=1)
            phi_l = np.take(phi_ext, range(0, n),   axis=1)
            if scheme == 'central':
                return (phi_r - phi_l) / (2.0 * dy)
            elif scheme == 'upwind':
                if c is None:
                    raise ValueError("Dy(..., scheme='upwind') requires c=velocity")
                c = _resolve(c)
                return np.where(c >= 0, (phi_c - phi_l) / dy,
                                        (phi_r - phi_c) / dy)
            else:
                raise ValueError(f"scheme must be 'central' or 'upwind'; got '{scheme}'")

        def Dxx(phi):
            phi    = _resolve(phi)
            n      = phi.shape[0]
            padded = _ghost_pad_dirichlet(phi, axis=0,
                         bc_lo=bct['x-'], bc_hi=bct['x+'],
                         val_lo=_bc_val('x-'), val_hi=_bc_val('x+'))
            r = np.take(padded, range(2, n+2), axis=0)
            c = np.take(padded, range(1, n+1), axis=0)
            l = np.take(padded, range(0, n),   axis=0)
            return (r - 2*c + l) / dx**2

        def Dyy(phi):
            phi    = _resolve(phi)
            n      = phi.shape[1]
            padded = _ghost_pad_dirichlet(phi, axis=1,
                         bc_lo=bct['y-'], bc_hi=bct['y+'],
                         val_lo=_bc_val('y-'), val_hi=_bc_val('y+'))
            r = np.take(padded, range(2, n+2), axis=1)
            c = np.take(padded, range(1, n+1), axis=1)
            l = np.take(padded, range(0, n),   axis=1)
            return (r - 2*c + l) / dy**2

        def Div_x(c, phi):
            """-d(c*phi)/dx, central differences."""
            cp  = _resolve(c) * _resolve(phi)
            n   = cp.shape[0]
            ext = _ghost_pad(cp, axis=0, bc_lo=bct['x-'], bc_hi=bct['x+'])
            r   = np.take(ext, range(2, n+2), axis=0)
            l   = np.take(ext, range(0, n),   axis=0)
            return -(r - l) / (2.0 * dx)

        def Div_y(c, phi):
            """-d(c*phi)/dy, central differences."""
            cp  = _resolve(c) * _resolve(phi)
            n   = cp.shape[1]
            ext = _ghost_pad(cp, axis=1, bc_lo=bct['y-'], bc_hi=bct['y+'])
            r   = np.take(ext, range(2, n+2), axis=1)
            l   = np.take(ext, range(0, n),   axis=1)
            return -(r - l) / (2.0 * dy)

        def Div_flux_x(k, phi):
            return _diffuse_2d(_resolve(phi), _resolve(k), dx, axis=0,
                               bc_lo=bct['x-'], bc_hi=bct['x+'])

        def Div_flux_y(k, phi):
            return _diffuse_2d(_resolve(phi), _resolve(k), dy, axis=1,
                               bc_lo=bct['y-'], bc_hi=bct['y+'])

    else:
        dx    = eq.dx
        bc_lo = 'periodic' if any(bc.kind == 'periodic' for bc in eq._bcs) \
                else 'none'
        bc_hi = bc_lo
        # get Dirichlet values for 1D
        val_lo = next((bc.get_value(0) for bc in eq._bcs
                       if bc.kind == 'dirichlet' and bc.mask[0]),  0.0)
        val_hi = next((bc.get_value(0) for bc in eq._bcs
                       if bc.kind == 'dirichlet' and bc.mask[-1]), 0.0)
        bc_lo_d = 'dirichlet' if any(
            bc.kind == 'dirichlet' and bc.mask[0]  for bc in eq._bcs) else bc_lo
        bc_hi_d = 'dirichlet' if any(
            bc.kind == 'dirichlet' and bc.mask[-1] for bc in eq._bcs) else bc_hi

        def Dx(phi, scheme='central', c=None):
            """∂phi/∂x. scheme='central' (default) or 'upwind' (requires c)."""
            phi = _resolve(phi).reshape(-1)
            p   = _ghost_pad(phi, 0, bc_lo, bc_hi)
            if scheme == 'central':
                return (p[2:] - p[:-2]) / (2.0 * dx)
            elif scheme == 'upwind':
                if c is None:
                    raise ValueError("Dx(..., scheme='upwind') requires c=velocity")
                c = _resolve(c).reshape(-1)
                return np.where(c >= 0, (phi - p[:-2]) / dx,
                                        (p[2:] - phi)  / dx)
            else:
                raise ValueError(f"scheme must be 'central' or 'upwind'; got '{scheme}'")

        def Dy(phi, scheme='central', c=None):
            raise NotImplementedError("Dy is not defined for 1D problems.")

        def Dxx(phi):
            p = _ghost_pad_dirichlet(_resolve(phi).reshape(-1), 0,
                                     bc_lo_d, bc_hi_d, val_lo, val_hi)
            return (p[2:] - 2*p[1:-1] + p[:-2]) / dx**2

        def Dyy(phi):
            raise NotImplementedError("Dyy is not defined for 1D problems.")

        def Div_x(c, phi):
            """-d(c*phi)/dx, central differences."""
            cp = _resolve(c) * _resolve(phi)
            p  = _ghost_pad(cp, 0, bc_lo, bc_hi)
            return -(p[2:] - p[:-2]) / (2.0 * dx)

        def Div_y(c, phi):
            raise NotImplementedError("Div_y is not defined for 1D problems.")

        def Div_flux_x(k, phi):
            return _diffuse_1d(_resolve(phi), _resolve(k), dx)

        def Div_flux_y(k, phi):
            raise NotImplementedError("Div_flux_y is not defined for 1D problems.")

    return {'Dx': Dx, 'Dy': Dy, 'Dxx': Dxx, 'Dyy': Dyy,
            'Div_x': Div_x, 'Div_y': Div_y,
            'Div_flux_x': Div_flux_x, 'Div_flux_y': Div_flux_y}



# ---------------------------------------------------------------------------
# Boundary condition descriptor
# ---------------------------------------------------------------------------

class _BC:
    """Internal representation of one boundary condition."""

    def __init__(self, kind, mask, value=None, normal=None):
        """
        kind   : 'periodic' | 'dirichlet' | 'neumann'
        mask   : bool ndarray matching the field shape
        value  : None | scalar | callable(t)
        normal : None | 'x-' | 'x+' | 'y-' | 'y+'
                 Direction of outward normal (2D Neumann only).
                 Inferred automatically from side= shortcuts.
        """
        self.kind   = kind
        self.mask   = mask
        self.value  = value
        self.normal = normal

    def get_value(self, t):
        if self.value is None:
            return None
        if callable(self.value):
            return self.value(t)
        return float(self.value)


# ---------------------------------------------------------------------------
# Interior obstacle descriptor
# ---------------------------------------------------------------------------

class _InteriorBC:
    """
    Descriptor for an interior region (obstacle / inclusion).

    kind    : 'dirichlet' — fix the field value inside the mask
              'neumann'   — approximate zero-flux: interior cells are frozen
                            (rhs=0). NOTE: this is a first-order approximation.
                            The fluid stencil at interface cells still reaches
                            into the frozen solid, so the no-flux condition is
                            only weakly enforced. Sufficient for pressure in NS
                            and coarse thermal insulation, but not accurate for
                            sharp flux-free boundaries.
    mask    : bool ndarray matching the field shape — True = interior/solid
    value   : scalar | callable(t) | callable(x[,y])  (Dirichlet only)
    """

    def __init__(self, kind, mask, value=None):
        if kind not in ('dirichlet', 'neumann'):
            raise ValueError(
                f"Interior BC kind must be 'dirichlet' or 'neumann'; got '{kind}'"
            )
        if kind == 'dirichlet' and value is None:
            raise ValueError("Interior Dirichlet BC requires a value.")
        self.kind  = kind
        self.mask  = np.asarray(mask, dtype=bool)
        self.value = value

    def get_value(self, t):
        """Evaluate prescribed value at time t (scalar or callable(t))."""
        if self.value is None:
            return 0.0
        if callable(self.value):
            try:
                return float(self.value(t))
            except TypeError:
                # value is a spatial callable — cannot evaluate at t alone;
                # return None to signal spatial evaluation needed
                return None



# ---------------------------------------------------------------------------
# Solution container
# ---------------------------------------------------------------------------

class PDESolution:
    """
    Result of PDESystem.solve().

    Attributes
    ----------
    t        : (nt,)         time points
    <field>  : (nx, nt)      1D solution array, accessed by name
               (nx, ny, nt)  2D solution array, accessed by name
    success  : bool
    message  : str
    raw      : original scipy solve_ivp result
    """

    def __init__(self, ivp_result, equations):
        self.t       = ivp_result.t
        self.success = ivp_result.success
        self.message = ivp_result.message
        self.raw     = ivp_result
        self._fields = [eq.field for eq in equations]
        nt           = len(ivp_result.t)

        offset = 0
        for eq in equations:
            size  = eq.n_total
            chunk = ivp_result.y[offset:offset + size, :]   # (size, nt)
            if eq.is_2d:
                # reshape to (nx, ny, nt)
                arr = chunk.reshape(eq.nx, eq.ny, nt)
            else:
                arr = chunk   # (nx, nt)
            setattr(self, eq.field, arr)
            offset += size

    def __repr__(self):
        f0    = self._fields[0]
        shape = getattr(self, f0).shape
        nt    = len(self.t)
        return (
            f"PDESolution(fields={self._fields}, shape={shape[:-1]}, nt={nt}, "
            f"t=[{self.t[0]:.3g}, {self.t[-1]:.3g}], success={self.success})"
        )


# ---------------------------------------------------------------------------
# PDE — single-field equation descriptor
# ---------------------------------------------------------------------------

class PDE:
    """
    Descriptor for a single PDE equation governing one field.

    Parameters
    ----------
    field : str
        Name of the unknown field, e.g. 'T' or 'cA'.
    x     : 1-D ndarray
        Grid in x (and only grid for 1D problems).
    y     : 1-D ndarray, optional
        Grid in y. If provided, the problem is treated as 2D.

    Notes
    -----
    PDE has no solve() method by design. Pass one or more PDE objects
    to PDESystem to solve them.

    In 2D, callable coefficients receive (x, y, **fields) where x and y
    are 2D meshgrid arrays of shape (nx, ny).
    """

    def __init__(self, field, x, y=None):
        if not isinstance(field, str):
            raise TypeError(
                "PDE() takes a single field name string as its first argument."
            )
        self.field = field
        self.x1d   = np.asarray(x, dtype=float)
        self.nx    = len(self.x1d)
        self.dx    = self.x1d[1] - self.x1d[0]

        if y is not None:
            self.y1d  = np.asarray(y, dtype=float)
            self.ny   = len(self.y1d)
            self.dy   = self.y1d[1] - self.y1d[0]
            self.is_2d = True
            # Build meshgrid: X[i,j] = x[i], Y[i,j] = y[j]  shape (nx, ny)
            self.X, self.Y = np.meshgrid(self.x1d, self.y1d, indexing='ij')
        else:
            self.y1d  = None
            self.ny   = 1
            self.dy   = None
            self.is_2d = False
            self.X    = self.x1d   # for 1D, coords tuple is just (x,)
            self.Y    = None

        self.n_total = self.nx * self.ny   # total DOFs for this field

        # keep .x and .n as 1D-compatible aliases
        self.x = self.X
        self.n = self.nx   # used by 1D BCs / legacy code

        self._advection    = []
        self._diffusion    = []
        self._sources      = []
        self._fluxes       = []
        self._generic_terms = []  # list of callables added via add_term()
        self._bcs          = []
        self._interior_bcs = []   # list of _InteriorBC
        self._ic           = None

    # ------------------------------------------------------------------
    # Public API: add terms
    # ------------------------------------------------------------------

    def add_advection(self, velocity=None, velocity_x=None, velocity_y=None):
        """
        Add convective advection term  c * d(phi)/d{x,y}  with first-order upwinding.

        The velocity c is the characteristic speed and sets the upwind direction
        via sign(c).  This is the strictly convective (non-conservative) form.
        Use add_flux() for conservation-law form  d(F(u))/dx.

        1D : add_advection(velocity=c)
             Adds  -c * dphi/dx   (upwind on sign(c))

        2D : add_advection(velocity_x=cx, velocity_y=cy)
             Adds  -cx * dphi/dx - cy * dphi/dy

        velocity / velocity_x / velocity_y may each be:
            scalar | ndarray | callable(x[,y], **fields) | string field name
        """
        if self.is_2d:
            if velocity_x is None and velocity_y is None:
                raise ValueError(
                    "In 2D use add_advection(velocity_x=..., velocity_y=...)"
                )
            self._advection.append({
                'velocity_x': _to_callable(velocity_x) if velocity_x is not None else None,
                'velocity_y': _to_callable(velocity_y) if velocity_y is not None else None,
            })
        else:
            if velocity is None:
                raise ValueError("In 1D use add_advection(velocity=...)")
            self._advection.append({'velocity': _to_callable(velocity)})
        return self

    def add_diffusion(self, diffusivity=None, diffusivity_x=None, diffusivity_y=None):
        """
        Add diffusion term(s).

        1D : add_diffusion(diffusivity=D)
             Adds  d(D·dphi/dx)/dx

        2D isotropic   : add_diffusion(diffusivity=D)
             Adds  d(D·dphi/dx)/dx + d(D·dphi/dy)/dy

        2D anisotropic : add_diffusion(diffusivity_x=Dx, diffusivity_y=Dy)
             Adds  d(Dx·dphi/dx)/dx + d(Dy·dphi/dy)/dy
        """
        if self.is_2d:
            if diffusivity is not None:
                # isotropic: same D in both directions
                D = _to_callable(diffusivity)
                self._diffusion.append({'diffusivity_x': D, 'diffusivity_y': D})
            elif diffusivity_x is not None or diffusivity_y is not None:
                self._diffusion.append({
                    'diffusivity_x': _to_callable(diffusivity_x) if diffusivity_x is not None else None,
                    'diffusivity_y': _to_callable(diffusivity_y) if diffusivity_y is not None else None,
                })
            else:
                raise ValueError("Provide diffusivity, diffusivity_x, or diffusivity_y")
        else:
            if diffusivity is None:
                raise ValueError("In 1D use add_diffusion(diffusivity=D)")
            self._diffusion.append({'diffusivity': _to_callable(diffusivity)})
        return self

    def add_source(self, expr):
        """
        Add a pointwise source term S(x[,y], **fields).

        expr : scalar | ndarray | callable(x, **fields)        [1D]
                                  callable(x, y, **fields)     [2D]
        """
        self._sources.append({'expr': _to_callable(expr)})
        return self

    def add_flux(self, flux=None, flux_x=None, flux_y=None, scheme='upwind'):
        """
        Add a conservation-law flux divergence  -dF/dx  (1D) or
        -dF_x/dx - dF_y/dy  (2D).

        This is the conservative form: use this when your equation is
            du/dt + dF(u)/dx = 0
        The upwind direction is determined by the local wave speed
            a(u) = dF/du
        estimated via finite difference, so sign(a) — not sign(F) — governs
        the stencil direction.  For the central scheme no wave speed is needed.

        1D : add_flux(flux=F)
             Adds  -dF(u)/dx

        2D : add_flux(flux_x=F, flux_y=G)
             Adds  -dF(u)/dx - dG(u)/dy
             Either flux_x or flux_y may be omitted.

        Parameters
        ----------
        flux   : callable(phi) -> ndarray   [1D]
        flux_x : callable(phi) -> ndarray   [2D, x-direction]
        flux_y : callable(phi) -> ndarray   [2D, y-direction]
        scheme : 'upwind' (default) | 'central'

        Notes
        -----
        For upwind scheme, each flux callable must accept the field array
        and return an array of the same shape.  The wave speed is inferred
        automatically as dF/dphi via forward finite difference (eps=1e-6*max|phi|).
        """
        if scheme not in ('upwind', 'central'):
            raise ValueError(
                f"scheme must be 'upwind' or 'central'; got '{scheme}'"
            )
        if self.is_2d:
            if flux_x is None and flux_y is None:
                raise ValueError(
                    "In 2D use add_flux(flux_x=..., flux_y=...)"
                )
            self._fluxes.append({
                'flux_x': _to_callable(flux_x) if flux_x is not None else None,
                'flux_y': _to_callable(flux_y) if flux_y is not None else None,
                'scheme': scheme,
            })
        else:
            if flux is None:
                raise ValueError("In 1D use add_flux(flux=F)")
            self._fluxes.append({'flux': _to_callable(flux), 'scheme': scheme})
        return self

    def add_term(self, fn):
        """
        Add a generic term to this equation's RHS using differential operators.

        fn : callable whose arguments are any combination of:
               - field names   (e.g. u, v, p, T)
               - Dx, Dy        central-difference d/dx, d/dy operators
               - Div_x, Div_y  central flux-divergence operators

             fn must return an ndarray of shape (nx,) [1D] or (nx,ny) [2D]
             representing the contribution to dfield/dt.

        The operators are BC-aware: ghost cells are padded according to
        each field's boundary conditions, so e.g. Dx(p) at a Dirichlet
        wall uses the correct ghost value, not a periodic wrap.

        Operator signatures
        -------------------
        Dx(phi)        d(phi)/dx  — central differences
        Dy(phi)        d(phi)/dy  — central differences
        Div_x(c, phi)  -d(c*phi)/dx  — 2nd-order central
        Div_y(c, phi)  -d(c*phi)/dy  — 2nd-order central

        phi and c may each be:
          - an ndarray of the right shape
          - a scalar  (broadcast to the grid)
          - a string field name  (looked up at eval time)

        Examples
        --------
        Navier-Stokes momentum (u-equation)::

            def u_rhs(u, v, p, Dx, Dy, Div_x, Div_y):
                return (- Div_x(u, u) - Div_y(v, u)
                        - (1/rho) * Dx(p)
                        + nu * (Dx(Dx(u)) + Dy(Dy(u))))
            eq_u.add_term(u_rhs)

        Pressure (artificial compressibility)::

            eq_p.add_term(lambda u, v, Dx, Dy: -beta * (Dx(u) + Dy(v)))

        Mixing with existing API (nonlinear diffusion + cross-field gradient)::

            eq_T.add_diffusion(diffusivity=lambda x, y, T: 1 + T**2)
            eq_T.add_term(lambda T, S, Dx: -alpha * Dx(S))
        """
        if not callable(fn):
            raise TypeError("add_term expects a callable.")
        self._generic_terms.append(fn)
        return self

    # ------------------------------------------------------------------
    # Public API: boundary conditions
    # ------------------------------------------------------------------

    def set_bc(self, kind, side=None, mask=None, value=None):
        """
        Set a boundary condition.

        kind  : 'periodic' | 'dirichlet' | 'neumann'
        side  : 1D : 'left' | 'right' | 'both'
                2D : 'left' | 'right' | 'bottom' | 'top' | 'x' | 'y' | 'all'
                     'x'   -- periodic in x  (left AND right walls)
                     'y'   -- periodic in y  (bottom AND top walls)
                     'all' -- periodic on all four sides
                (shorthand; ignored if mask= is given)
        mask  : bool ndarray matching the field shape (nx,) or (nx, ny)
                Overrides side=.
        value : scalar | callable(t)
                dirichlet -- prescribed field value
                neumann   -- prescribed outward normal derivative dphi/dn

        Notes
        -----
        Periodic BCs must be set in *pairs* -- a boundary is periodic only if
        both ends of that axis are periodic.  Use side='x', side='y', or
        side='all' to express this clearly in 2D.  In 1D, any call with
        kind='periodic' is treated as fully periodic (both ends).

        Examples (2D)::

            eq.set_bc(kind='periodic', side='x')   # periodic left/right
            eq.set_bc(kind='periodic', side='y')   # periodic bottom/top
            eq.set_bc(kind='periodic', side='all') # fully periodic
        """
        if kind == 'periodic':
            if self.is_2d:
                if side is None or side == 'all':
                    axes = ['x', 'y']
                elif side in ('x', 'left', 'right'):
                    axes = ['x']
                elif side in ('y', 'bottom', 'top'):
                    axes = ['y']
                else:
                    raise ValueError(
                        f"For periodic 2D BCs use side='x', 'y', or 'all'; got '{side}'"
                    )
                dummy = np.zeros((self.nx, self.ny), dtype=bool)
                for ax in axes:
                    self._bcs.append(_BC('periodic', dummy, normal=ax))
            else:
                self._bcs.append(_BC('periodic', np.zeros(self.nx, dtype=bool)))
            return self

        if kind not in ('dirichlet', 'neumann'):
            raise ValueError(
                f"kind must be 'periodic', 'dirichlet', or 'neumann'; got '{kind}'"
            )
        if value is None:
            raise ValueError(f"value is required for '{kind}' BC")

        if mask is not None:
            bc_mask = np.asarray(mask, dtype=bool)
            normal  = None   # user-supplied mask; normal inferred later if needed
        elif side is not None:
            bc_mask, normal = self._side_to_mask(side)
        else:
            raise ValueError(
                "Either 'side' or 'mask' must be provided for non-periodic BCs"
            )

        self._bcs.append(_BC(kind, bc_mask, value, normal=normal))
        return self

    def set_interior_bc(self, mask, kind='dirichlet', value=None):
        """
        Mark an interior region as a solid obstacle or inclusion.

        This implements the inactive-cell approach: the RHS at all masked
        points is forced to zero (or dvalue/dt for time-varying Dirichlet)
        at every time step, so the integrator never evolves those cells.
        Neighbouring active cells see the obstacle's fixed values through
        the stencil, which imposes an approximate interface condition.

        Parameters
        ----------
        mask  : bool ndarray of shape (nx,) [1D] or (nx, ny) [2D]
                True where the obstacle/inclusion occupies the grid.
        kind  : 'dirichlet' (default) | 'neumann'
                'dirichlet' — fix the field to *value* inside the mask.
                              Use for no-slip walls (value=0), heated
                              cylinders (value=T_wall), etc.
                'neumann'   — zero-flux approximation.  The interior cells
                              are frozen at their IC values and the stencil
                              at interface cells naturally sees those fixed
                              values, giving an approximate no-flux wall.
                              value= is ignored.
        value : scalar | callable(t) | None
                Prescribed field value inside the obstacle (Dirichlet only).
                callable(t) is evaluated at each time step, enabling
                time-varying interior temperatures / concentrations.

        Notes
        -----
        * Interior BCs are applied *after* all stencil terms and after
          domain-edge BCs, so they always win at masked points.
        * The IC at masked points is automatically snapped to *value* at
          t=t0 when PDESystem.solve() is called, ensuring consistency.
        * For Neumann ('no-flux') obstacles, initialise the field inside
          the mask to the ambient value you want the wall to hold; uPDE
          will keep it frozen there.
        * Multiple calls accumulate — you can place several obstacles of
          different kinds on the same field.

        Examples
        --------
        Heated cylinder (Dirichlet, T_wall = 2.0)::

            mask = (X - cx)**2 + (Y - cy)**2 < r**2
            eq.set_interior_bc(mask, kind='dirichlet', value=2.0)

        Insulating obstacle (zero-flux, frozen at ambient T=0)::

            eq.set_interior_bc(mask, kind='neumann')
            # initialise IC to 0 everywhere, including inside mask

        Pulsating heat source::

            eq.set_interior_bc(mask, kind='dirichlet',
                               value=lambda t: 1.0 + 0.5*np.sin(2*np.pi*t))
        """
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != ((self.nx, self.ny) if self.is_2d else (self.nx,)):
            raise ValueError(
                f"Interior BC mask shape {mask.shape} does not match field "
                f"shape {(self.nx, self.ny) if self.is_2d else (self.nx,)}."
            )
        self._interior_bcs.append(_InteriorBC(kind, mask, value))
        return self

    # ------------------------------------------------------------------
    # Public API: initial condition
    # ------------------------------------------------------------------

    def set_ic(self, ic):
        """
        Set the initial condition.

        ic : scalar
               | ndarray of shape (nx,) [1D] or (nx, ny) [2D]
               | callable:
                   1D: ic(x)       -> ndarray (nx,)
                   2D: ic(x, y)    -> ndarray (nx, ny)   [x,y are meshgrid arrays]

        Boundary points are snapped to Dirichlet BC values at solve time.
        """
        self._ic = ic
        return self

    def _resolve_ic(self):
        """Evaluate the stored IC to a flat numpy array of shape (n_total,)."""
        ic = self._ic
        if ic is None:
            return None
        if callable(ic):
            if self.is_2d:
                arr = np.asarray(ic(self.X, self.Y), dtype=float)
            else:
                arr = np.asarray(ic(self.x1d), dtype=float)
        elif np.isscalar(ic):
            arr = np.full(self.n_total, float(ic))
            return arr
        else:
            arr = np.asarray(ic, dtype=float)
        return arr.ravel()

    # ------------------------------------------------------------------
    # Introspection (used by PDESystem for validation)
    # ------------------------------------------------------------------

    def field_refs(self):
        """Return all field names referenced by any callable in this equation."""
        refs = set()
        for term in self._advection:
            for key in ('velocity', 'velocity_x', 'velocity_y'):
                fn = term.get(key)
                if fn is None:
                    continue
                if hasattr(fn, '_field_ref_name'):
                    refs.add(fn._field_ref_name)
                else:
                    refs |= _callable_field_refs(fn)
        for term in self._diffusion:
            for key in ('diffusivity', 'diffusivity_x', 'diffusivity_y'):
                if term.get(key) is not None:
                    refs |= _callable_field_refs(term[key])
        for term in self._sources:
            refs |= _callable_field_refs(term['expr'])
        for term in self._fluxes:
            for key in ('flux', 'flux_x', 'flux_y'):
                if term.get(key) is not None:
                    refs |= _callable_field_refs(term[key])
        for fn in self._generic_terms:
            refs |= _callable_field_refs(fn)
        return refs

    # ------------------------------------------------------------------
    # Internals
    def solve(self, t_span, **kwargs):
        """
        Solve this equation directly without constructing a PDESystem manually.

        Raises ValueError if the equation references external fields — in that
        case it is coupled and must be solved via PDESystem explicitly.

        Parameters
        ----------
        t_span : (t0, tf)
        **kwargs : forwarded to PDESystem.solve() / solve_ivp
                   (method, t_eval, rtol, atol, dense_output, ...)

        Returns
        -------
        PDESolution

        Example
        -------
        eq = HeatEquation('T', x=x, diffusivity=0.01)
        eq.set_bc(side='left',  kind='dirichlet', value=1.0)
        eq.set_bc(side='right', kind='dirichlet', value=0.0)
        eq.set_ic(0.0)
        sol = eq.solve(t_span=(0, 1), method='BDF')
        sol.T   # shape (nx, nt)
        """
        # External refs are any field names other than this equation's own field
        external_refs = self.field_refs() - {self.field}
        if external_refs:
            raise ValueError(
                f"PDE('{self.field}') references external field(s) {sorted(external_refs)} "
                f"and cannot be solved independently. "
                f"Use PDESystem([...]).solve() to solve coupled equations together."
            )
        return PDESystem([self]).solve(t_span, **kwargs)

    # ------------------------------------------------------------------

    def _side_to_mask(self, side):
        """Return (mask, normal_direction) for a named side."""
        if self.is_2d:
            mask = np.zeros((self.nx, self.ny), dtype=bool)
            normals = {
                'left':   ('x-', lambda m: m.__setitem__((0,  slice(None)), True)),
                'right':  ('x+', lambda m: m.__setitem__((-1, slice(None)), True)),
                'bottom': ('y-', lambda m: m.__setitem__((slice(None),  0), True)),
                'top':    ('y+', lambda m: m.__setitem__((slice(None), -1), True)),
            }
            if side not in normals:
                raise ValueError(
                    f"In 2D, side must be 'left','right','bottom','top'; got '{side}'"
                )
            normal, setter = normals[side]
            setter(mask)
            return mask, normal
        else:
            mask   = np.zeros(self.nx, dtype=bool)
            normal_map = {
                'left':  'x-', 'right': 'x+', 'both': None
            }
            if side == 'left':
                mask[0]  = True;  return mask, 'x-'
            elif side == 'right':
                mask[-1] = True;  return mask, 'x+'
            elif side == 'both':
                mask[0]  = True
                mask[-1] = True;  return mask, None
            else:
                raise ValueError(
                    f"In 1D, side must be 'left', 'right', or 'both'; got '{side}'"
                )

    def _coords(self):
        """Return coordinate tuple for _call: (x,) in 1D, (X, Y) in 2D."""
        if self.is_2d:
            return (self.X, self.Y)
        return (self.x1d,)

    def _bc_types_2d(self):
        """
        Return the BC kind for each of the four 2D edges.

        Returns a dict:
            {'x-': kind, 'x+': kind, 'y-': kind, 'y+': kind}
        where kind is 'periodic', 'dirichlet', 'neumann', or 'none'.

        Used by _ghost_pad to select ghost values per axis-end.
        """
        result = {'x-': 'none', 'x+': 'none', 'y-': 'none', 'y+': 'none'}
        for bc in self._bcs:
            if bc.kind == 'periodic':
                # bc.normal is 'x' or 'y' (set by new set_bc logic)
                if bc.normal == 'x':
                    result['x-'] = result['x+'] = 'periodic'
                elif bc.normal == 'y':
                    result['y-'] = result['y+'] = 'periodic'
                else:
                    # legacy / fully-periodic fallback
                    result['x-'] = result['x+'] = 'periodic'
                    result['y-'] = result['y+'] = 'periodic'
            elif bc.normal in result:
                result[bc.normal] = bc.kind
        return result
    def __repr__(self):
        dim  = f"2D ({self.nx}×{self.ny})" if self.is_2d else f"1D (n={self.nx})"
        bcs  = [bc.kind for bc in self._bcs]
        return (
            f"PDE('{self.field}', {dim}, dx={self.dx:.4g}  |  "
            f"{len(self._advection)} advection, "
            f"{len(self._diffusion)} diffusion, "
            f"{len(self._sources)} source  |  BCs: {bcs})"
        )


# ---------------------------------------------------------------------------
# PDESystem — couples PDE descriptors and owns the solver
# ---------------------------------------------------------------------------

class PDESystem:
    """
    Couple a list of PDE descriptors into a solvable system.

    Parameters
    ----------
    equations : list of PDE

    Raises at construction time
    ---------------------------
    TypeError  — if any element is not a PDE
    ValueError — duplicate field names
    ValueError — grid size mismatch between equations
    ValueError — any callable references an undeclared field
    """

    def __init__(self, equations):
        for eq in equations:
            if not isinstance(eq, PDE):
                raise TypeError(
                    f"PDESystem expects PDE objects; got {type(eq)}"
                )

        fields = [eq.field for eq in equations]
        seen   = set()
        for f in fields:
            if f in seen:
                raise ValueError(f"Duplicate field '{f}' in PDESystem")
            seen.add(f)

        n_total = equations[0].n_total
        for eq in equations:
            if eq.n_total != n_total:
                raise ValueError(
                    f"Grid size mismatch: '{eq.field}' has {eq.n_total} DOFs, "
                    f"expected {n_total}"
                )

        field_set = set(fields)
        for eq in equations:
            unknown = eq.field_refs() - field_set
            if unknown:
                raise ValueError(
                    f"Equation for '{eq.field}' references undeclared "
                    f"field(s): {unknown}.  Declared fields: {fields}"
                )

        self.equations = equations
        self.fields    = fields
        self.n_total   = n_total

    # ------------------------------------------------------------------
    # Public API: solve
    # ------------------------------------------------------------------

    def solve(self, t_span, ICs=None, method='RK45', t_eval=None, **kwargs):
        """
        Solve the coupled PDE system.

        Parameters
        ----------
        t_span : (t0, tf)
        ICs    : dict, optional
                 field -> initial condition (scalar | ndarray | callable)
                 Overrides or supplements ICs set via eq.set_ic().
                 callable: 1D → ic(x), 2D → ic(x, y) with meshgrid arrays
        method : str passed to solve_ivp (default 'RK45')
        t_eval : optional array of output times
        **kwargs : forwarded to solve_ivp (rtol, atol, max_step, ...)

        Returns
        -------
        PDESolution
          1D fields: shape (nx, nt)
          2D fields: shape (nx, ny, nt)
        """
        ICs = dict(ICs) if ICs is not None else {}

        # Normalise ICs passed in dict
        for f, val in ICs.items():
            eq = self._eq_by_field(f)
            if callable(val):
                if eq.is_2d:
                    ICs[f] = np.asarray(val(eq.X, eq.Y), dtype=float).ravel()
                else:
                    ICs[f] = np.asarray(val(eq.x1d), dtype=float).ravel()
            elif np.isscalar(val):
                ICs[f] = np.full(eq.n_total, float(val))
            else:
                ICs[f] = np.asarray(val, dtype=float).ravel()

        # Fill in per-equation ICs where not overridden
        for eq in self.equations:
            if eq.field not in ICs:
                arr = eq._resolve_ic()
                if arr is None:
                    raise ValueError(
                        f"No initial condition for field '{eq.field}'. "
                        f"Use eq.set_ic() or pass ICs={{'{eq.field}': ...}} to solve()."
                    )
                ICs[eq.field] = arr

        # Snap Dirichlet boundary points to their t=t0 values.
        # In 2D, apply y-normal BCs (bottom/top) first, then x-normal BCs
        # (left/right) so that corners are owned by the x-direction walls.
        # This gives a consistent IC that matches what _apply_bcs enforces
        # during time integration.
        t0 = t_span[0]
        for eq in self.equations:
            ic_2d = ICs[eq.field].reshape(eq.nx, eq.ny) if eq.is_2d \
                    else ICs[eq.field]
            if eq.is_2d:
                # Pass 1: y-normal walls (bottom/top)
                for bc in eq._bcs:
                    if bc.kind == 'dirichlet' and bc.normal in ('y-', 'y+', None):
                        ic_2d[bc.mask] = bc.get_value(t0)
                # Pass 2: x-normal walls (left/right) — overwrites corners
                for bc in eq._bcs:
                    if bc.kind == 'dirichlet' and bc.normal in ('x-', 'x+'):
                        ic_2d[bc.mask] = bc.get_value(t0)
            else:
                for bc in eq._bcs:
                    if bc.kind == 'dirichlet':
                        ic_2d[bc.mask] = bc.get_value(t0)
            # Snap interior Dirichlet BCs (obstacles / inclusions)
            for ibc in eq._interior_bcs:
                if ibc.kind == 'dirichlet':
                    v = ibc.get_value(t0)
                    if v is not None:
                        ic_2d[ibc.mask] = v
            ICs[eq.field] = ic_2d.ravel()

        y0  = np.concatenate([ICs[f] for f in self.fields])
        ivp = solve_ivp(
            self._rhs, t_span, y0,
            method=method, t_eval=t_eval, **kwargs
        )
        return PDESolution(ivp, self.equations)

    # ------------------------------------------------------------------
    # Internal: RHS for solve_ivp
    # ------------------------------------------------------------------

    def _rhs(self, t, y):
        # Unpack flat y into per-field arrays (2D or 1D)
        fields_flat = {}   # flat  (n_total,)
        fields_nd   = {}   # shaped (nx,) or (nx,ny)
        offset = 0
        for eq in self.equations:
            chunk = y[offset:offset + eq.n_total]
            fields_flat[eq.field] = chunk
            fields_nd[eq.field]   = chunk.reshape(eq.nx, eq.ny) if eq.is_2d \
                                    else chunk
            offset += eq.n_total

        rhs_nd = {eq.field: np.zeros((eq.nx, eq.ny) if eq.is_2d else (eq.nx,))
                  for eq in self.equations}

        for eq in self.equations:
            phi    = fields_nd[eq.field]
            coords = eq._coords()
            rhs    = rhs_nd[eq.field]

            # --- Advection: convective form  -c * dphi/d{x,y}, upwind on sign(c) ---
            bct = eq._bc_types_2d() if eq.is_2d else {}
            for term in eq._advection:
                if eq.is_2d:
                    if term['velocity_x'] is not None:
                        cx  = _call(term['velocity_x'], coords, fields_nd)
                        rhs -= _convect_2d(phi, cx, eq.dx, axis=0,
                                           bc_lo=bct['x-'], bc_hi=bct['x+'])
                    if term['velocity_y'] is not None:
                        cy  = _call(term['velocity_y'], coords, fields_nd)
                        rhs -= _convect_2d(phi, cy, eq.dy, axis=1,
                                           bc_lo=bct['y-'], bc_hi=bct['y+'])
                else:
                    c     = _call(term['velocity'], coords, fields_nd)
                    bc_lo = 'periodic' if any(bc.kind=='periodic' for bc in eq._bcs) else 'none'
                    bc_hi = bc_lo
                    rhs  -= _convect_1d(phi, c, eq.dx, bc_lo, bc_hi)

            # --- Diffusion ---
            for term in eq._diffusion:
                if eq.is_2d:
                    if term['diffusivity_x'] is not None:
                        Dx   = _call(term['diffusivity_x'], coords, fields_nd)
                        rhs += _diffuse_2d(phi, Dx, eq.dx, axis=0,
                                           bc_lo=bct['x-'], bc_hi=bct['x+'])
                    if term['diffusivity_y'] is not None:
                        Dy   = _call(term['diffusivity_y'], coords, fields_nd)
                        rhs += _diffuse_2d(phi, Dy, eq.dy, axis=1,
                                           bc_lo=bct['y-'], bc_hi=bct['y+'])
                else:
                    D    = _call(term['diffusivity'], coords, fields_nd)
                    rhs += _diffuse_1d(phi, D, eq.dx)

            # --- Sources ---
            for term in eq._sources:
                rhs += _call(term['expr'], coords, fields_nd)

            # --- Conservation-law fluxes: -dF/dx [-dG/dy] ---
            # flux callables take the field array directly: F(phi) -> ndarray
            for term in eq._fluxes:
                if eq.is_2d:
                    if term.get('flux_x') is not None:
                        fn = term['flux_x']
                        F  = fn(phi)
                        if term['scheme'] == 'central':
                            rhs -= _central_flux_2d(F, eq.dx, axis=0,
                                                    bc_lo=bct['x-'], bc_hi=bct['x+'])
                        else:
                            a = _wave_speed(fn, phi)
                            rhs -= _upwind_flux_2d(F, a, eq.dx, axis=0,
                                                   bc_lo=bct['x-'], bc_hi=bct['x+'])
                    if term.get('flux_y') is not None:
                        fn = term['flux_y']
                        F  = fn(phi)
                        if term['scheme'] == 'central':
                            rhs -= _central_flux_2d(F, eq.dy, axis=1,
                                                    bc_lo=bct['y-'], bc_hi=bct['y+'])
                        else:
                            a = _wave_speed(fn, phi)
                            rhs -= _upwind_flux_2d(F, a, eq.dy, axis=1,
                                                   bc_lo=bct['y-'], bc_hi=bct['y+'])
                else:
                    fn = term['flux']
                    F  = fn(phi)
                    if term['scheme'] == 'central':
                        rhs -= _central_flux_1d(F, eq.dx)
                    else:
                        a = _wave_speed(fn, phi)
                        rhs -= _upwind_flux_1d(F, a, eq.dx)

            # --- Generic add_term callables ---
            if eq._generic_terms:
                ops = _build_operators(eq, fields_nd, bct if eq.is_2d else {})
                for fn in eq._generic_terms:
                    sig    = inspect.signature(fn)
                    params = sig.parameters
                    has_var_kw = any(
                        p.kind == inspect.Parameter.VAR_KEYWORD
                        for p in params.values()
                    )
                    if has_var_kw:
                        # Inject all operators and all fields
                        kwargs = {**ops, **fields_nd}
                    else:
                        kwargs = {}
                        for name in params:
                            if name in fields_nd:
                                kwargs[name] = fields_nd[name]
                            elif name in ops:
                                kwargs[name] = ops[name]
                    rhs += fn(**kwargs)

            # --- Domain-edge boundary conditions ---
            rhs = _apply_bcs(eq, rhs, phi, t)

            # --- Interior obstacle / inclusion BCs ---
            # Applied last so they always override stencil values at
            # masked (solid) points, regardless of what the stencil computed.
            for ibc in eq._interior_bcs:
                if ibc.kind == 'dirichlet':
                    v = ibc.get_value(t)
                    if v is not None:
                        # Constant or time-varying scalar value:
                        # set dφ/dt = 0 (constant) or dvalue/dt (time-varying)
                        if callable(ibc.value):
                            dt_fd = 1e-8
                            dvdt  = (ibc.value(t + dt_fd) - ibc.value(t)) / dt_fd
                            rhs[ibc.mask] = dvdt
                        else:
                            rhs[ibc.mask] = 0.0
                    else:
                        # Spatial callable value(x[,y]): freeze at IC value
                        rhs[ibc.mask] = 0.0
                elif ibc.kind == 'neumann':
                    # Approximate zero-flux: freeze interior cells.
                    # Interface fluid cells still see the frozen solid value
                    # in their stencil — a first-order approximation sufficient
                    # for pressure obstacles and coarse insulation.
                    rhs[ibc.mask] = 0.0

            rhs_nd[eq.field] = rhs

        return np.concatenate([rhs_nd[eq.field].ravel() for eq in self.equations])

    # ------------------------------------------------------------------

    def _eq_by_field(self, field):
        for eq in self.equations:
            if eq.field == field:
                return eq
        raise KeyError(field)

    def __repr__(self):
        lines = [f"PDESystem({len(self.equations)} equations)"]
        for eq in self.equations:
            lines.append(f"  {eq}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Discretisation — module-level, independently testable
# ---------------------------------------------------------------------------

def _upwind_1d(phi, c, dx):
    """First-order upwind -d(c·phi)/dx in 1D."""
    phi_l = np.roll(phi,  1)
    phi_r = np.roll(phi, -1)
    F     = c * phi
    F_l   = np.roll(c, 1) * phi_l
    dFdx  = np.where(
        c >= 0,
        (F - F_l) / dx,
        (np.roll(c, -1) * phi_r - F) / dx,
    )
    return -dFdx


def _ghost_pad(arr, axis, bc_lo, bc_hi):
    """
    Pad arr by one ghost cell on each end along axis.

    Ghost value choice:
      'periodic'              -- wrap from opposite end
      'dirichlet'/'neumann'/'none' -- replicate boundary value
        (zero-gradient ghost; Dirichlet rows are overwritten by _apply_bcs,
         Neumann flux correction is applied there too)
    """
    lo = np.take(arr, [-1] if bc_lo == 'periodic' else [0],  axis=axis)
    hi = np.take(arr, [0]  if bc_hi == 'periodic' else [-1], axis=axis)
    return np.concatenate([lo, arr, hi], axis=axis)


def _ghost_pad_dirichlet(arr, axis, bc_lo, bc_hi, val_lo=0.0, val_hi=0.0):
    """
    Ghost-cell padding with proper Dirichlet ghost values.

    For Dirichlet boundaries, the ghost is set so that the average of
    ghost and boundary equals the prescribed value:
        ghost = 2*value - boundary_node
    This gives correct second-derivative stencils at Dirichlet walls.

    Used by Dxx/Dyy operators in add_term callables.
    """
    if bc_lo == 'periodic':
        lo = np.take(arr, [-1], axis=axis)
    elif bc_lo == 'dirichlet':
        bnd = np.take(arr, [0], axis=axis)
        lo  = 2.0 * val_lo - bnd
    else:
        lo = np.take(arr, [0], axis=axis)   # zero-gradient (Neumann/none)

    if bc_hi == 'periodic':
        hi = np.take(arr, [0], axis=axis)
    elif bc_hi == 'dirichlet':
        bnd = np.take(arr, [-1], axis=axis)
        hi  = 2.0 * val_hi - bnd
    else:
        hi = np.take(arr, [-1], axis=axis)

    return np.concatenate([lo, arr, hi], axis=axis)


def _upwind_2d(phi, c, d, axis, bc_lo='none', bc_hi='none'):
    """First-order upwind -d(c*phi)/d{axis} in 2D with ghost-cell padding."""
    n       = phi.shape[axis]
    phi_ext = _ghost_pad(phi, axis, bc_lo, bc_hi)
    c_ext   = _ghost_pad(c,   axis, bc_lo, bc_hi)

    c_c   = np.take(c_ext,   range(1, n+1), axis=axis)
    phi_c = np.take(phi_ext, range(1, n+1), axis=axis)
    phi_r = np.take(phi_ext, range(2, n+2), axis=axis)
    phi_l = np.take(phi_ext, range(0, n),   axis=axis)
    c_r   = np.take(c_ext,   range(2, n+2), axis=axis)
    c_l   = np.take(c_ext,   range(0, n),   axis=axis)

    F_c = c_c * phi_c
    F_l = c_l * phi_l
    dFd = np.where(
        c_c >= 0,
        (F_c - F_l) / d,
        (c_r * phi_r - F_c) / d,
    )
    return -dFd


def _wave_speed(flux_fn, phi, eps=1e-6):
    """
    Estimate local wave speed a(phi) = dF/dphi via forward finite difference.
    eps is scaled by max|phi| to handle fields of varying magnitude.
    """
    scale = np.max(np.abs(phi))
    h     = eps * scale if scale > 0 else eps
    return (flux_fn(phi + h) - flux_fn(phi)) / h


def _convect_1d(phi, c, dx, bc_lo='none', bc_hi='none'):
    """
    Convective form  c * dphi/dx  with first-order upwinding on sign(c).
    Returns the term as-is (caller negates: rhs -= _convect_1d(...)).
    """
    p        = _ghost_pad(phi, 0, bc_lo, bc_hi)
    dphi_bwd = (phi - p[:-2]) / dx   # backward difference  (phi[i] - phi[i-1])
    dphi_fwd = (p[2:] - phi)  / dx   # forward difference   (phi[i+1] - phi[i])
    return c * np.where(c >= 0, dphi_bwd, dphi_fwd)


def _convect_2d(phi, c, d, axis, bc_lo='none', bc_hi='none'):
    """
    Convective form  c * dphi/d{axis}  with first-order upwinding on sign(c).
    """
    n       = phi.shape[axis]
    phi_ext = _ghost_pad(phi, axis, bc_lo, bc_hi)
    phi_c   = np.take(phi_ext, range(1, n+1), axis=axis)
    phi_r   = np.take(phi_ext, range(2, n+2), axis=axis)
    phi_l   = np.take(phi_ext, range(0, n),   axis=axis)
    dphi_bwd = (phi_c - phi_l) / d
    dphi_fwd = (phi_r - phi_c) / d
    return c * np.where(c >= 0, dphi_bwd, dphi_fwd)


def _upwind_flux_1d(F, a, dx):
    """
    Upwind flux differencing  dF/dx  in 1D.
    Upwind direction determined by wave speed a = dF/dphi (not sign(F)).
    Uses simple donor-cell: F_{i+1/2} = F_i if a>=0 else F_{i+1}.
    """
    F_l  = np.roll(F,  1)   # F[i-1]
    F_r  = np.roll(F, -1)   # F[i+1]
    # face flux at i+1/2: upwind on a
    F_iph = np.where(a >= 0, F,   F_r)   # right face
    F_imh = np.where(a >= 0, F_l, F  )   # left face
    return (F_iph - F_imh) / dx


def _upwind_flux_2d(F, a, d, axis, bc_lo='none', bc_hi='none'):
    """
    Upwind flux differencing  dF/d{axis}  in 2D.
    Upwind direction determined by wave speed a = dF/dphi.
    """
    n     = F.shape[axis]
    F_ext = _ghost_pad(F, axis, bc_lo, bc_hi)
    a_ext = _ghost_pad(a, axis, bc_lo, bc_hi)
    F_c   = np.take(F_ext, range(1, n+1), axis=axis)
    F_r   = np.take(F_ext, range(2, n+2), axis=axis)
    F_l   = np.take(F_ext, range(0, n),   axis=axis)
    a_c   = np.take(a_ext, range(1, n+1), axis=axis)
    F_iph = np.where(a_c >= 0, F_c, F_r)   # right face
    F_imh = np.where(a_c >= 0, F_l, F_c)   # left face
    return (F_iph - F_imh) / d


def _central_flux_1d(F, dx):
    """Central differencing of flux  dF/dx  in 1D."""
    return (np.roll(F, -1) - np.roll(F, 1)) / (2.0 * dx)


def _central_flux_2d(F, d, axis, bc_lo='none', bc_hi='none'):
    """Central differencing of flux  dF/d{axis}  in 2D."""
    n     = F.shape[axis]
    F_ext = _ghost_pad(F, axis, bc_lo, bc_hi)
    F_r   = np.take(F_ext, range(2, n+2), axis=axis)
    F_l   = np.take(F_ext, range(0, n),   axis=axis)
    return (F_r - F_l) / (2.0 * d)


def _diffuse_1d(phi, D, dx):
    """Conservative central d(D·dphi/dx)/dx in 1D."""
    D_r   = 0.5 * (D + np.roll(D, -1))
    D_l   = 0.5 * (D + np.roll(D,  1))
    phi_r = np.roll(phi, -1)
    phi_l = np.roll(phi,  1)
    return (D_r * (phi_r - phi) - D_l * (phi - phi_l)) / dx**2


def _diffuse_2d(phi, D, d, axis, bc_lo='none', bc_hi='none'):
    """Conservative central d(D*dphi/d{axis})/d{axis} in 2D with ghost-cell padding."""
    n       = phi.shape[axis]
    phi_ext = _ghost_pad(phi, axis, bc_lo, bc_hi)
    D_ext   = _ghost_pad(D,   axis, bc_lo, bc_hi)

    phi_c = np.take(phi_ext, range(1, n+1), axis=axis)
    phi_r = np.take(phi_ext, range(2, n+2), axis=axis)
    phi_l = np.take(phi_ext, range(0, n),   axis=axis)
    D_c   = np.take(D_ext,   range(1, n+1), axis=axis)
    D_r   = 0.5 * (D_c + np.take(D_ext, range(2, n+2), axis=axis))
    D_l   = 0.5 * (D_c + np.take(D_ext, range(0, n),   axis=axis))

    return (D_r * (phi_r - phi_c) - D_l * (phi_c - phi_l)) / d**2


def _apply_bcs(eq, rhs, phi, t):
    """
    Apply all BCs to the assembled rhs.

    Dirichlet: overwrite rhs at BC nodes with dvalue/dt (0 for constant BCs).
    Neumann 1D: the _diffuse_1d stencil used np.roll (periodic ghost); correct
                the boundary node RHS here using the proper one-sided ghost.
    Neumann 2D: the _diffuse_2d stencil used a zero-gradient ghost (replicates
                the boundary value), which gives zero normal flux by default.
                Correct the boundary row here for any non-zero Neumann value.
    """
    if eq.is_2d:
        # --- Neumann 2D corrections ---
        for bc in eq._bcs:
            if bc.kind == 'neumann':
                g = bc.get_value(t)
                if g == 0.0:
                    continue   # zero-gradient ghost was already correct
                normal = bc.normal or _infer_normal_2d(bc.mask, phi.shape)
                if normal == 'x-':
                    # rhs[0,:] was computed with ghost=phi[0,:] (zero flux).
                    # Correct: ghost = phi[1,:] - 2*g*dx  (outward normal = -x)
                    # Delta = D_l*(phi[0]-phi_ghost_correct - phi[0]+phi_ghost_used)/dx^2
                    #       = D_l*2*g*dx/dx^2 = 2*D*g/dx  (use D[0,:] as approx)
                    # Simpler: just recompute rhs at the boundary row properly.
                    rhs[0, :][bc.mask[0, :]] += (
                        2.0 * g / eq.dx * np.ones(eq.ny)[bc.mask[0, :]]
                    )
                elif normal == 'x+':
                    rhs[-1, :][bc.mask[-1, :]] -= (
                        2.0 * g / eq.dx * np.ones(eq.ny)[bc.mask[-1, :]]
                    )
                elif normal == 'y-':
                    rhs[:, 0][bc.mask[:, 0]] += (
                        2.0 * g / eq.dy * np.ones(eq.nx)[bc.mask[:, 0]]
                    )
                elif normal == 'y+':
                    rhs[:, -1][bc.mask[:, -1]] -= (
                        2.0 * g / eq.dy * np.ones(eq.nx)[bc.mask[:, -1]]
                    )

        # --- Dirichlet 2D: y-walls first, x-walls second (corners -> x wins) ---
        for bc in eq._bcs:
            if bc.kind == 'dirichlet' and bc.normal in ('y-', 'y+', None):
                _set_dirichlet_rhs(rhs, bc, t)
        for bc in eq._bcs:
            if bc.kind == 'dirichlet' and bc.normal in ('x-', 'x+'):
                _set_dirichlet_rhs(rhs, bc, t)

    else:
        # --- 1D ---
        for bc in eq._bcs:
            if bc.kind == 'dirichlet':
                _set_dirichlet_rhs(rhs, bc, t)
            elif bc.kind == 'neumann':
                # _diffuse_1d used np.roll (periodic ghost); fix boundary nodes.
                g  = bc.get_value(t)
                dx = eq.dx
                if bc.mask[0]:     # left:  outward normal = -x  =>  -dphi/dx = g
                    ghost    = phi[1] - 2.0 * g * dx
                    rhs[0]   = (phi[1] - phi[0] - (phi[0] - ghost)) / dx**2
                if bc.mask[-1]:    # right: outward normal = +x  =>  +dphi/dx = g
                    ghost    = phi[-2] + 2.0 * g * dx
                    rhs[-1]  = (ghost - phi[-1] - (phi[-1] - phi[-2])) / dx**2

    return rhs


def _set_dirichlet_rhs(rhs, bc, t):
    """Set rhs at Dirichlet BC nodes to dvalue/dt (0 for constant)."""
    if callable(bc.value):
        dt_fd = 1e-8
        dvdt  = (bc.value(t + dt_fd) - bc.value(t)) / dt_fd
        rhs[bc.mask] = dvdt
    else:
        rhs[bc.mask] = 0.0


def _infer_normal_2d(mask, shape):
    """
    Infer outward normal from which edge the masked points lie on.
    Raises if the mask spans multiple edges (ambiguous).
    """
    on_left   = np.any(mask[0,  :])
    on_right  = np.any(mask[-1, :])
    on_bottom = np.any(mask[:,  0])
    on_top    = np.any(mask[:, -1])
    edges = [on_left, on_right, on_bottom, on_top]
    if sum(edges) != 1:
        raise ValueError(
            "Cannot infer Neumann normal: mask spans multiple edges. "
            "Use set_bc(side=...) or set_bc(mask=...) with one edge at a time."
        )
    return ['x-', 'x+', 'y-', 'y+'][edges.index(True)]
