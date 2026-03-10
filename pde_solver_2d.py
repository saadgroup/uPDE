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
    """
    if callable(val):
        return val
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
        f"Coefficient must be scalar, ndarray, or callable; got {type(val)}"
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
    (not 'x', 'y', 't', and not *args / **kwargs).
    """
    if not callable(fn):
        return set()
    sig    = inspect.signature(fn)
    ignore = {'x', 'y', 't'}
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
              'neumann'   — zero-flux approximation (inactive cells:
                            RHS zeroed, neighbours see the fixed value
                            at their stencil interface)
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
        return float(self.value)


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
        self._bcs          = []
        self._interior_bcs = []   # list of _InteriorBC
        self._ic           = None

    # ------------------------------------------------------------------
    # Public API: add terms
    # ------------------------------------------------------------------

    def add_advection(self, velocity=None, velocity_x=None, velocity_y=None):
        """
        Add advection term(s).

        1D : add_advection(velocity=...)
             Adds  -d(c·phi)/dx

        2D : add_advection(velocity_x=..., velocity_y=...)
             Adds  -d(cx·phi)/dx - d(cy·phi)/dy
             velocity_x and velocity_y may each be scalar|ndarray|callable(x,y,**fields)
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

    def add_flux(self, flux, scheme='upwind'):
        """
        Escape-hatch: add a custom flux -dF/dx with explicit scheme.
        1D only for now; 2D custom fluxes require flux_x and flux_y.

        flux   : callable(x, **fields) -> ndarray
        scheme : 'upwind' | 'central'
        """
        if scheme not in ('upwind', 'central'):
            raise ValueError(
                f"scheme must be 'upwind' or 'central'; got '{scheme}'"
            )
        self._fluxes.append({'flux': _to_callable(flux), 'scheme': scheme})
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
                if term.get(key) is not None:
                    refs |= _callable_field_refs(term[key])
        for term in self._diffusion:
            for key in ('diffusivity', 'diffusivity_x', 'diffusivity_y'):
                if term.get(key) is not None:
                    refs |= _callable_field_refs(term[key])
        for term in self._sources:
            refs |= _callable_field_refs(term['expr'])
        for term in self._fluxes:
            refs |= _callable_field_refs(term['flux'])
        return refs

    # ------------------------------------------------------------------
    # Internals
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

            # --- Advection ---
            if eq.is_2d:
                bct = eq._bc_types_2d()
            for term in eq._advection:
                if eq.is_2d:
                    if term['velocity_x'] is not None:
                        cx   = _call(term['velocity_x'], coords, fields_nd)
                        rhs += _upwind_2d(phi, cx, eq.dx, axis=0,
                                          bc_lo=bct['x-'], bc_hi=bct['x+'])
                    if term['velocity_y'] is not None:
                        cy   = _call(term['velocity_y'], coords, fields_nd)
                        rhs += _upwind_2d(phi, cy, eq.dy, axis=1,
                                          bc_lo=bct['y-'], bc_hi=bct['y+'])
                else:
                    c    = _call(term['velocity'], coords, fields_nd)
                    rhs += _upwind_1d(phi, c, eq.dx)

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

            # --- Custom fluxes (1D only for now) ---
            for term in eq._fluxes:
                F = _call(term['flux'], coords, fields_nd)
                if term['scheme'] == 'central':
                    rhs -= np.gradient(F, eq.dx)
                else:
                    rhs += _upwind_flux_1d(F, eq.dx)

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
                    # Zero-flux / insulating: simply freeze interior cells.
                    # The stencil at interface cells already sees the fixed
                    # interior values (set in IC), giving approximate no-flux.
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


def _upwind_flux_1d(F, dx):
    """Upwind differencing for a precomputed flux F (1D)."""
    F_l  = np.roll(F, 1)
    dFdx = np.where(
        F >= 0,
        (F - F_l) / dx,
        (np.roll(F, -1) - F) / dx,
    )
    return -dFdx


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
