"""
chemistry.py
============
Flamelet-based thermochemistry for uPDE mixture-fraction transport.

The central object is :class:`FlameletTable`, which maps mixture fraction
Z ∈ [0, 1] to temperature T and species mass fractions Y_k via
pre-integrated 1-D counterflow flame profiles.  At solve time the table
is just a numpy interpolation call — zero additional stiffness is added
to the PDE system.

Construction paths
------------------
1. **Analytic Burke-Schumann** (no external dependencies)::

       table = FlameletTable.burke_schumann(
           fuel='CH4', Z_st=0.055,
           T_fuel=300.0, T_ox=300.0, T_ad=2230.0
       )

2. **From saved file** (numpy .npz, no Cantera needed at solve time)::

       table = FlameletTable.from_file('ch4_air_flamelet.npz')

3. **From raw arrays** (user supplies their own data)::

       table = FlameletTable(Z_grid, T, species={'CH4': Y_CH4, 'O2': Y_O2})

4. **From Cantera** (requires ``pip install cantera``; generates a true
   counterflow diffusion flame and extracts the Z-T-Y profiles)::

       table = FlameletTable.from_cantera(
           mechanism='gri30.yaml',
           fuel='CH4',
           oxidizer='O2:0.21,N2:0.79',
           T_fuel=300.0, T_ox=300.0,
           pressure=101325.0,
           n_points=200,
       )
       table.save('ch4_air_flamelet.npz')

Usage at solve time
-------------------
All accessors accept any numpy array of shape matching the spatial grid::

    T_field   = table.T(sol.Z[:, -1])          # 1-D: shape (nx,)
    T_field   = table.T(sol.Z[:, :, -1])        # 2-D: shape (nx, ny)
    Y_CH4     = table.Y('CH4', sol.Z[:, -1])
    rho_field = table.rho(sol.Z[:, -1], P=101325.0)

Z values are automatically clipped to [0, 1] before interpolation.
"""

import numpy as np


# ---------------------------------------------------------------------------
# FlameletTable
# ---------------------------------------------------------------------------

class FlameletTable:
    """
    Pre-tabulated flamelet thermochemistry: Z → T, Y_k.

    Parameters
    ----------
    Z_grid  : 1-D array, shape (n,), values in [0, 1], monotonically increasing
    T       : 1-D array, shape (n,)   — temperature [K]
    species : dict[str, 1-D array]    — {name: Y_k array of shape (n,)}
    MW      : dict[str, float] or None — molecular weights [kg/kmol];
              used only by :meth:`rho`. If None, density is unavailable.

    All arrays are copied on construction so the table is immutable after
    creation.
    """

    # Default molecular weights [kg/kmol] for common combustion species
    _MW_DEFAULT = {
        'CH4':  16.04,
        'O2':   32.00,
        'CO2':  44.01,
        'H2O':  18.02,
        'N2':   28.01,
        'CO':   28.01,
        'H2':    2.02,
        'OH':   17.01,
        'NO':   30.01,
        'AR':   39.95,
    }

    def __init__(self, Z_grid, T, species, MW=None):
        Z_grid = np.asarray(Z_grid, dtype=float)
        T      = np.asarray(T,      dtype=float)
        if Z_grid.ndim != 1 or T.ndim != 1:
            raise ValueError("Z_grid and T must be 1-D arrays.")
        if Z_grid.shape != T.shape:
            raise ValueError("Z_grid and T must have the same length.")
        if not np.all(np.diff(Z_grid) > 0):
            raise ValueError("Z_grid must be strictly increasing.")
        if Z_grid[0] < 0 or Z_grid[-1] > 1:
            raise ValueError("Z_grid values must lie in [0, 1].")

        self._Z  = Z_grid.copy()
        self._T  = T.copy()
        self._Y  = {k: np.asarray(v, dtype=float).copy()
                    for k, v in species.items()}
        for k, v in self._Y.items():
            if v.shape != self._Z.shape:
                raise ValueError(
                    f"Species '{k}' array shape {v.shape} != Z_grid shape {self._Z.shape}."
                )

        # Molecular weights: merge user-supplied with defaults
        self._MW = dict(self._MW_DEFAULT)
        if MW is not None:
            self._MW.update(MW)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def species(self):
        """List of species names stored in the table."""
        return list(self._Y.keys())

    @property
    def Z_grid(self):
        """The mixture-fraction grid, shape (n,)."""
        return self._Z.copy()

    @property
    def Z_st(self):
        """
        Stoichiometric mixture fraction: Z where T is maximum.
        Returns the Z value of peak temperature.
        """
        return float(self._Z[np.argmax(self._T)])

    def T(self, Z):
        """
        Interpolate temperature [K] at mixture fraction Z.

        Parameters
        ----------
        Z : array-like — mixture fraction, clipped to [0, 1]

        Returns
        -------
        ndarray, same shape as Z
        """
        Z = np.clip(np.asarray(Z, dtype=float), self._Z[0], self._Z[-1])
        return np.interp(Z, self._Z, self._T)

    def Y(self, species_name, Z):
        """
        Interpolate species mass fraction Y_k at mixture fraction Z.

        Parameters
        ----------
        species_name : str — must be in self.species
        Z            : array-like — mixture fraction, clipped to [0, 1]

        Returns
        -------
        ndarray, same shape as Z
        """
        if species_name not in self._Y:
            raise KeyError(
                f"Species '{species_name}' not in table. "
                f"Available: {self.species}"
            )
        Z = np.clip(np.asarray(Z, dtype=float), self._Z[0], self._Z[-1])
        return np.interp(Z, self._Z, self._Y[species_name])

    def rho(self, Z, P=101325.0, R_u=8314.0):
        """
        Mixture density [kg/m³] from ideal-gas law: ρ = P * W_mix / (R_u * T).

        Requires molecular weights to be available (either via MW_DEFAULT
        or the MW argument to the constructor) for all species in the table.

        Parameters
        ----------
        Z   : array-like — mixture fraction
        P   : float      — pressure [Pa]   (default 1 atm)
        R_u : float      — universal gas constant [J/kmol/K]

        Returns
        -------
        ndarray, same shape as Z
        """
        Z  = np.clip(np.asarray(Z, dtype=float), self._Z[0], self._Z[-1])
        T  = self.T(Z)

        # mean molecular weight: W_mix = 1 / sum(Y_k / W_k)
        inv_W = np.zeros_like(Z)
        for k, Y_arr in self._Y.items():
            if k not in self._MW:
                raise KeyError(
                    f"Molecular weight for species '{k}' not available. "
                    f"Pass MW={{'{k}': value}} to the constructor."
                )
            inv_W += np.interp(Z, self._Z, Y_arr) / self._MW[k]
        W_mix = 1.0 / np.where(inv_W > 0, inv_W, 1e-30)
        return P * W_mix / (R_u * T)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path):
        """
        Save the table to a numpy .npz file.

        Parameters
        ----------
        path : str — file path (the .npz extension is added if absent)
        """
        arrays = {'Z_grid': self._Z, 'T': self._T}
        for k, v in self._Y.items():
            arrays[f'Y_{k}'] = v
        np.savez(path, **arrays)

    @classmethod
    def from_file(cls, path):
        """
        Load a FlameletTable from a .npz file saved by :meth:`save`.

        Parameters
        ----------
        path : str — path to .npz file

        Returns
        -------
        FlameletTable
        """
        data    = np.load(path)
        Z_grid  = data['Z_grid']
        T       = data['T']
        species = {k[2:]: data[k] for k in data.files
                   if k.startswith('Y_')}
        return cls(Z_grid, T, species)

    # ------------------------------------------------------------------
    # Analytic table: Burke-Schumann equilibrium flame
    # ------------------------------------------------------------------

    @classmethod
    def burke_schumann(cls,
                       fuel='CH4',
                       Z_st=0.055,
                       T_fuel=300.0,
                       T_ox=300.0,
                       T_ad=2230.0,
                       n_points=500):
        """
        Analytic Burke-Schumann (infinitely fast chemistry) flamelet table
        for methane-air combustion.  No external dependencies required.

        The temperature profile is piecewise linear in Z, peaking at T_ad
        at the stoichiometric mixture fraction Z_st.  Species mass fractions
        follow piecewise linear profiles consistent with complete combustion:

            CH4 + 2 O2 → CO2 + 2 H2O

        Nitrogen is treated as an inert diluent in air (Y_N2 = 0.767 on
        the oxidizer side, diluted linearly with Z).

        Parameters
        ----------
        fuel      : str   — fuel label (stored as metadata; only 'CH4'
                            stoichiometry is implemented)
        Z_st      : float — stoichiometric mixture fraction (default 0.055
                            for CH4/air at 300 K, 1 atm)
        T_fuel    : float — fuel-stream temperature [K]  (default 300 K)
        T_ox      : float — oxidizer-stream temperature [K]  (default 300 K)
        T_ad      : float — adiabatic flame temperature [K]  (default 2230 K)
        n_points  : int   — number of Z points in the table

        Returns
        -------
        FlameletTable

        Notes
        -----
        The Burke-Schumann solution is exact for one-step infinitely-fast
        chemistry.  It places the "flame" as an infinitely thin sheet at
        Z = Z_st.  Real flames are broader due to finite-rate chemistry and
        differential diffusion; use :meth:`from_cantera` for quantitative work.

        Default Z_st = 0.055 corresponds to stoichiometric CH4/air
        (mass-based: 1 kg CH4 needs ~17.2 kg air, Z_st = 1/18.2 ≈ 0.055).
        """
        Z = np.linspace(0.0, 1.0, n_points)

        # --- Temperature: piecewise linear, peak at Z_st ---
        T = np.where(
            Z <= Z_st,
            T_ox  + (T_ad - T_ox)  * (Z / Z_st),
            T_ad  + (T_fuel - T_ad) * ((Z - Z_st) / (1.0 - Z_st)),
        )

        # --- Species mass fractions (CH4/air, complete combustion) ---
        # Stoichiometry: CH4 + 2 O2 → CO2 + 2 H2O
        # MW: CH4=16.04, O2=32.00, CO2=44.01, H2O=18.02, N2=28.01
        # Air: Y_O2=0.233, Y_N2=0.767 (mass fractions)
        #
        # Fuel side (Z=1): Y_CH4=1, all others=0
        # Ox  side (Z=0): Y_O2=0.233, Y_N2=0.767, all others=0
        #
        # Lean side (Z < Z_st): CH4 completely consumed
        #   Y_CH4 = 0
        #   Y_O2  = 0.233*(1-Z) - (32/16.04)*Z       (O2 consumed by CH4)
        #           clamped to 0 at Z=Z_st
        #   Y_CO2 = (44.01/16.04)*Z                   (CO2 produced)
        #   Y_H2O = (2*18.02/16.04)*Z                 (H2O produced)
        #   Y_N2  = 0.767*(1-Z)
        #
        # Rich side (Z > Z_st): O2 completely consumed
        #   Y_CH4 = Z - Z_st*(16.04/(2*32.00))*(1-Z)/(1-Z_st) [approx]
        #           simpler: linear from 0 at Z_st to 1 at Z=1
        #   Y_O2  = 0
        #   Y_CO2 and Y_H2O follow O2-limited production

        nu_O2  = 2.0 * 32.00 / 16.04    # kg O2 per kg CH4
        nu_CO2 = 44.01 / 16.04           # kg CO2 per kg CH4
        nu_H2O = 2.0 * 18.02 / 16.04    # kg H2O per kg CH4

        Y_O2_air = 0.233
        Y_N2_air = 0.767

        # --- Lean side (Z <= Z_st) ---
        Z_lean = np.minimum(Z, Z_st)
        Y_CH4_lean = np.zeros(n_points)
        Y_O2_lean  = np.maximum(Y_O2_air * (1.0 - Z) - nu_O2 * Z, 0.0)
        Y_CO2_lean = nu_CO2 * Z_lean
        Y_H2O_lean = nu_H2O * Z_lean
        Y_N2_lean  = Y_N2_air * (1.0 - Z)

        # --- Rich side (Z > Z_st) ---
        # Available O2 at the stoichiometric point
        Z_rich = np.maximum(Z, Z_st)
        # Fraction of fuel burned = limited by available O2
        # At Z_st all O2 is consumed: Y_O2 = 0 for Z >= Z_st
        Z_hat = (Z - Z_st) / (1.0 - Z_st + 1e-30)   # 0 at Z_st, 1 at Z=1
        Y_CH4_rich = Z_hat                             # linear 0→1
        Y_O2_rich  = np.zeros(n_points)
        # CO2 and H2O produced are limited by O2 available at Z_st
        # Amount burned at Z_st point scales with oxidiser available in mixture
        Y_CO2_rich = nu_CO2 * Z_st * (1.0 - Z_hat)
        Y_H2O_rich = nu_H2O * Z_st * (1.0 - Z_hat)
        Y_N2_rich  = Y_N2_air * (1.0 - Z_rich)

        # --- Blend lean/rich with step at Z_st ---
        lean = Z <= Z_st
        Y_CH4 = np.where(lean, Y_CH4_lean, Y_CH4_rich)
        Y_O2  = np.where(lean, Y_O2_lean,  Y_O2_rich)
        Y_CO2 = np.where(lean, Y_CO2_lean, Y_CO2_rich)
        Y_H2O = np.where(lean, Y_H2O_lean, Y_H2O_rich)
        Y_N2  = np.where(lean, Y_N2_lean,  Y_N2_rich)

        # Normalise so sum of Y_k = 1 everywhere (enforce conservation)
        Y_sum = Y_CH4 + Y_O2 + Y_CO2 + Y_H2O + Y_N2
        Y_CH4 /= Y_sum
        Y_O2  /= Y_sum
        Y_CO2 /= Y_sum
        Y_H2O /= Y_sum
        Y_N2  /= Y_sum

        species = {
            'CH4': Y_CH4,
            'O2':  Y_O2,
            'CO2': Y_CO2,
            'H2O': Y_H2O,
            'N2':  Y_N2,
        }
        return cls(Z, T, species)

    # ------------------------------------------------------------------
    # Cantera-based table generation (optional dependency)
    # ------------------------------------------------------------------

    @classmethod
    def from_cantera(cls,
                     mechanism='gri30.yaml',
                     fuel='CH4',
                     oxidizer='O2:0.21,N2:0.79',
                     T_fuel=300.0,
                     T_ox=300.0,
                     pressure=101325.0,
                     width=0.02,
                     n_points=200,
                     loglevel=0):
        """
        Generate a FlameletTable from a Cantera counterflow diffusion flame.

        Requires ``pip install cantera``.  The flame is solved on a 1-D domain
        with fuel on the right and oxidiser on the left; the mixture fraction
        is computed at every grid point using Bilger's definition, and the
        $Z$-$T$-$Y_k$ profiles are extracted, sorted, and interpolated onto
        a uniform Z grid.

        Parameters
        ----------
        mechanism : str   — Cantera mechanism file, e.g. 'gri30.yaml'
        fuel      : str   — fuel species name, e.g. 'CH4'
        oxidizer  : str   — oxidizer composition, e.g. 'O2:0.21,N2:0.79'
        T_fuel    : float — fuel-stream temperature [K]
        T_ox      : float — oxidizer-stream temperature [K]
        pressure  : float — pressure [Pa]
        width     : float — domain width [m] (default 2 cm)
        n_points  : int   — number of Z points in the output table
        loglevel  : int   — Cantera verbosity (0 = silent)

        Returns
        -------
        FlameletTable

        Raises
        ------
        ImportError
            If Cantera is not installed.

        Notes
        -----
        The mixture fraction is computed via Bilger's definition:

            Z = (β - β_ox) / (β_fuel - β_ox)

        where β = 2*Z_C/W_C + 0.5*Z_H/W_H - Z_O/W_O  (elemental mass
        fractions divided by atomic weights).

        Example
        -------
        table = FlameletTable.from_cantera(
            mechanism='gri30.yaml',
            fuel='CH4',
            oxidizer='O2:0.21,N2:0.79',
        )
        table.save('ch4_air_gri30.npz')
        """
        try:
            import cantera as ct
        except ImportError:
            raise ImportError(
                "Cantera is required for FlameletTable.from_cantera(). "
                "Install it with:  pip install cantera\n"
                "Alternatively, use FlameletTable.burke_schumann() for an "
                "analytic approximation that needs no extra dependencies."
            )

        # --- Set up fuel and oxidizer streams ---
        gas = ct.Solution(mechanism)

        gas.TPX = T_fuel, pressure, f'{fuel}:1.0'
        gas_fuel = gas.X.copy()

        gas.TPX = T_ox, pressure, oxidizer
        gas_ox = gas.X.copy()

        # --- Solve counterflow diffusion flame ---
        f = ct.CounterflowDiffusionFlame(gas, width=width)
        f.fuel_inlet.mdot     = 0.24      # kg/m²/s — typical value
        f.fuel_inlet.X        = gas_fuel
        f.fuel_inlet.T        = T_fuel
        f.oxidizer_inlet.mdot = 0.72
        f.oxidizer_inlet.X    = gas_ox
        f.oxidizer_inlet.T    = T_ox
        f.set_refine_criteria(ratio=3, slope=0.1, curve=0.2, prune=0.02)
        f.solve(loglevel=loglevel, auto=True)

        # --- Compute Bilger mixture fraction ---
        # Atomic weights [kg/kmol]
        W_C, W_H, W_O = 12.011, 1.008, 15.999

        # Elemental mass fractions at each grid point
        Z_C = np.array([
            sum(gas.atomic_weight('C') * gas.n_atoms(sp, 'C') * f.Y[gas.species_index(sp), :]
                for sp in gas.species_names)
        ]).squeeze() if 'C' in [e for e in gas.element_names] else np.zeros(len(f.grid))

        # Simpler: use Cantera's built-in mixture fraction if available (ct >= 3.0)
        try:
            Z_bilger = f.mixture_fraction(fuel, oxidizer)
        except AttributeError:
            # Fallback: compute manually for C-H-O system
            Y_all = f.Y   # shape (n_species, n_points)
            beta  = np.zeros(Y_all.shape[1])
            for sp_idx, sp_name in enumerate(gas.species_names):
                sp = gas.species(sp_idx)
                n_C = sp.composition.get('C', 0)
                n_H = sp.composition.get('H', 0)
                n_O = sp.composition.get('O', 0)
                MW  = gas.molecular_weights[sp_idx]
                beta += Y_all[sp_idx] * (
                    2.0 * n_C / (W_C * MW) * MW +
                    0.5 * n_H / (W_H * MW) * MW -
                    1.0 * n_O / (W_O * MW) * MW
                ) / MW * gas.molecular_weights[sp_idx]

            # Recompute at pure fuel and pure oxidizer
            gas.TPX = T_fuel, pressure, f'{fuel}:1.0'
            beta_fuel = sum(
                gas.Y[gas.species_index(sp)] * (
                    2.0 * gas.species(gas.species_index(sp)).composition.get('C', 0) / W_C +
                    0.5 * gas.species(gas.species_index(sp)).composition.get('H', 0) / W_H -
                    1.0 * gas.species(gas.species_index(sp)).composition.get('O', 0) / W_O
                )
                for sp in gas.species_names
            )
            gas.TPX = T_ox, pressure, oxidizer
            beta_ox  = sum(
                gas.Y[gas.species_index(sp)] * (
                    2.0 * gas.species(gas.species_index(sp)).composition.get('C', 0) / W_C +
                    0.5 * gas.species(gas.species_index(sp)).composition.get('H', 0) / W_H -
                    1.0 * gas.species(gas.species_index(sp)).composition.get('O', 0) / W_O
                )
                for sp in gas.species_names
            )
            Z_bilger = (beta - beta_ox) / (beta_fuel - beta_ox + 1e-30)

        Z_bilger = np.clip(Z_bilger, 0.0, 1.0)
        T_flame  = f.T
        Y_flame  = f.Y   # shape (n_species, n_points)

        # --- Sort by Z (flame profile is not monotone in Z in general) ---
        order    = np.argsort(Z_bilger)
        Z_sorted = Z_bilger[order]
        T_sorted = T_flame[order]
        Y_sorted = Y_flame[:, order]

        # Remove duplicate Z values (keep last = highest T at stoich)
        _, unique_idx = np.unique(Z_sorted, return_index=True)
        Z_sorted = Z_sorted[unique_idx]
        T_sorted = T_sorted[unique_idx]
        Y_sorted = Y_sorted[:, unique_idx]

        # --- Interpolate onto uniform Z grid ---
        Z_uniform = np.linspace(0.0, 1.0, n_points)
        T_out     = np.interp(Z_uniform, Z_sorted, T_sorted)
        species   = {}
        for sp_idx, sp_name in enumerate(gas.species_names):
            Y_sp = np.interp(Z_uniform, Z_sorted, Y_sorted[sp_idx])
            if np.any(Y_sp > 1e-10):   # skip trace species
                species[sp_name] = Y_sp

        MW = {sp: gas.molecular_weights[gas.species_index(sp)]
              for sp in species}

        return cls(Z_uniform, T_out, species, MW=MW)

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self):
        return (
            f"FlameletTable(n={len(self._Z)}, "
            f"Z=[{self._Z[0]:.3f},{self._Z[-1]:.3f}], "
            f"T=[{self._T[0]:.0f},{self._T.max():.0f}]K, "
            f"species={self.species})"
        )
