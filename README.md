# emergentintensity

`emergentintensity` is a Python package to compute the emergent Stokes I and Q normalized by B (Planck function) from a plane-parallel slab using precomputed radiative-transfer tables with smooth interpolation and thin/thick patches.

## Citation

If you use this code or the tables, please cite:
- Kitade & Kataoka 2026


## Installation

```bash
git clone https://github.com/astroaki/emergentintensity.git
cd emergentintensity
pip install -e .
```

After installation, you can launch notebooks by:

```bash
pip install jupyter
jupyter lab
```

## Quick start

```python
from emergentintensity import emergent_stokes

tau_max = 0.1
omega   = 0.8
inc_deg = 60.0

I, Q = emergent_stokes(tau_max, omega, inc_deg)
print("I/B =", I)
print("Q/B =", Q)
print("Q/I =", Q/I if I != 0 else 0.0)
```

## Usage

Main API:
- `emergent_stokes(tau_max, omega, inc_deg) -> (I_over_B, Q_over_B)`
- Polarization fraction can be computed as `PF = Q/I`.

Inputs:
- `tau_max`: total vertical extinction optical depth
- `omega`  : single-scattering albedo
- `inc_deg`: inclination in degrees

The package reads RT tables from `data/` (e.g., `taumax*_I_emergent.inp`, `taumax*_Q_emergent.inp`) and thin-regime Q table `Q_table.inp`.

## Valid parameter ranges

This package is based on precomputed RT tables. Please keep inputs within the tabulated domain.

- Albedo `omega`
  - RT tables are provided only up to `omega = 0.9`.
  - Do not use `omega > 0.9` (results are not validated).

- Inclination `inc_deg`:
  - use up to 89.0 degree.

- total vertical extinction optical depth `tau_max`:
  - RT tables are provided for `tau_max` in [0.01, 15].
  - For `tau_max < 0.01`, we apply optically-thin patches:
    - Stokes I is computed using the analytic thin form
      `I/B = 1 - exp(-tau_abs / mu)` with `tau_abs = (1 - omega) * tau_max`.
    - Stokes Q uses a precomputed thin-regime table (from Eq. (B.10), file `Q_table.inp`)
      for `tau_max` in [1e-4, 1e-2] (with interpolation in `mu`, `omega`, and `tau_max`).
      For `tau_max < 1e-4`, Q is extrapolated using an `~tau_max^2` form Eq.(B.4),
      with the coefficient chosen to match the table smoothly at `tau_max = 1e-4`.
  - For `tau_max > 15`, Stokes I and Q are saturated by clamping to the RT-table values at `tau_max = 15`
    (i.e., we return the same values as `tau_max = 15`).  



## Notebooks

Example notebooks in [`notebooks/`](notebooks/):

- [`exampleusage.ipynb`](notebooks/exampleusage.ipynb): basic usage
- [`plot_emergent_intensity.ipynb`](notebooks/plot_emergent_intensity.ipynb): plotting the Stokes I, Q, and polarization fraction
- [`plot_regimes.ipynb`](notebooks/plot_regimes.ipynb): regime map
- [`plot_fig3.ipynb`](notebooks/plot_fig3.ipynb): Fig.3 reproduction; omega dependence of I, Q, and polarization fraction
- [`plot_fig4.ipynb`](notebooks/plot_fig4.ipynb): Fig.4 reproduction; the peak polarization fraction as a function of inclination
- [`plot_fig9.ipynb`](notebooks/plot_fig9.ipynb): Fig.9 reproduction; comparison with analytic formulae
- [`plot_fig11-14.ipynb`](notebooks/plot_fig11-14.ipynb): Fig.11-14 reproduction; comparison between fitting formulae and RT results


## Requirements
- Python >= 3.9
- NumPy
- Scipy
- Matplotlib (only for plotting examples/notebooks)


## Repository layout
- `src/emergentintensity.py` : main code
- `data/` : RT tables (`taumax*_I_emergent.inp`, `taumax*_Q_emergent.inp`) and `Q_table.inp`
- `notebooks/` : usage & plotting examples

By default, the code searches tables in data/ (repository root).


## License
MIT