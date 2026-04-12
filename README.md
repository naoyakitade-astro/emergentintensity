# emergentintensity

This repository provides three Python modules for radiative-transfer-table-based calculations and fitting:

## Modules

- `stokes_interpolation.py`: computes emergent Stokes I and Q, and polarization fractions by interpolating precomputed radiative-transfer tables
- `stokes_i_fitting.py`: fits the emergent Stokes I using fitting functions
- `pf_fitting.py`: fits the polarization fraction using fitting functions


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

## Examples

### stokes_interpolation.py

```python
from stokes_interpolation import emergent_stokes

I, Q = emergent_stokes(1.0, 0.5, 45.0)
print(I, Q)
```

### stokes_i_fitting.py

```python
from stokes_i_fitting import fit_and_plot_I

result = fit_and_plot_I(inc=45.0)
```

### pf_fitting.py

```python
from pf_fitting import fit_and_plot_PF

result = fit_and_plot_PF(inc=45.0) 
```

## Data files

The modules read radiative-transfer tables from `data/`, including:

- `StokesI_emergent/`
- `StokesQ_emergent/`
- `Q_table.inp`


## Valid parameter ranges

### stokes_interpolation.py

- Albedo `omega`
  - RT tables are provided only up to `omega = 0.9`.
  - Do not use `omega > 0.9` (results are not validated).

- Inclination `inc`:
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

### stokes_i_fitting.py

- Albedo `omega`: `0.0 < omega < 0.9`  
  Values with `omega >= 0.9` are not validated.

- Inclination `inc_deg`: `inc_deg <= 80.0` degrees

- Total vertical extinction optical depth `tau_max`: `tau_max >= 0.0`

### pf_fitting.py

- Albedo `omega`: `0.0 < omega < 0.9`  
  Values with `omega >= 0.9` are not validated.

- Inclination `inc_deg`: `inc_deg <= 80.0` degrees

- Total vertical extinction optical depth `tau_max`: `tau_max >= 0.0`



## Notebooks

Example notebooks in [`notebooks/`](notebooks/):

- [`stokes_interpolation_example.ipynb`](notebooks/stokes_interpolation_example.ipynb): basic usage of `stokes_interpolation.py`
- [`pf_fitting_example.ipynb`](notebooks/pf_fitting_example.ipynb): example for `pf_fitting.py`
- [`stokes_i_fitting_example.ipynb`](notebooks/stokes_i_fitting_example.ipynb): example for `stokes_i_fitting.py`


## Requirements

- Python >= 3.9
- NumPy
- SciPy
- Matplotlib
- lmfit


## Repository layout

- `src/stokes_interpolation.py`
- `src/pf_fitting.py`
- `src/stokes_i_fitting.py`
- `data/`
- `notebooks/`


## License
MIT