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
from stokes_interpolation import emergent_stokes, setup_tables

context = setup_tables(data_dir="data")

I, Q = emergent_stokes(
    tau_max=1.0,
    omega=0.5,
    inc=45.0,
    context=context,
)
print(I, Q)
```

`emergent_stokes()` also accepts `inc_deg=` as a backward-compatible alias for `inc=`.

### stokes_i_fitting.py

```python
from stokes_i_fitting import fit_and_plot_I, load_I_only_inp

data = load_I_only_inp()
result = fit_and_plot_I(inc=45.0, data=data)
```

### pf_fitting.py

```python
from pf_fitting import fit_and_plot_PF, load_IQ_inp

data = load_IQ_inp()
result = fit_and_plot_PF(inc=45.0, data=data)
```

## Data files

The modules read radiative-transfer tables from `data/`, including:

- `StokesI_emergent/`
- `StokesQ_emergent/`
- `Q_table.inp`

## Valid parameter ranges

Inputs outside the validated user range now raise `ValueError` so that failures are explicit.

### stokes_interpolation.py

- Albedo `omega`
  - validated range: `0.0 <= omega <= 0.9`
  - `omega > 0.9` raises `ValueError`
  - For `omega = 0`, the code uses the analytic pure-absorption solution rather than interpolating the RT tables.
  - For `omega > 0`, the emergent Stokes parameters are obtained from the numerical RT tables, and the interpolation in `omega` is constructed so that the solution connects smoothly from the analytic `omega = 0` limit to the first tabulated numerical solution at `omega = 0.1`.
- Inclination `inc`
  - validated range: `0.0 <= inc <= 89.0` degrees
- Total vertical extinction optical depth `tau_max`
  - validated range: `tau_max >= 0.0`
  - RT tables are provided for `tau_max` in `[0.01, 15]`
  - For `tau_max < 0.01`, we apply optically-thin patches:
    - Stokes I is computed using the analytic thin form
      `I/B = 1 - exp(-tau_abs / mu)` with `tau_abs = (1 - omega) * tau_max`
    - Stokes Q uses a precomputed thin-regime table (file `Q_table.inp`)
      for `tau_max` in `[1e-4, 1e-2]`
  - For `tau_max > 15`, Stokes I and Q are saturated by clamping to the RT-table values at `tau_max = 15`, and a `RuntimeWarning` is emitted

### stokes_i_fitting.py

- Albedo `omega`
  - validated range for the fitted evaluator: `0.0 <= omega <= 0.9`
- Inclination `inc`
  - validated range for fitting: `0.0 <= inc <= 80.0` degrees
- Total vertical extinction optical depth `tau_max`
  - validated range: `tau_max >= 0.0`

### pf_fitting.py

- Albedo `omega`
  - validated range for the fitted evaluator: `0.0 < omega <= 0.9`
- Inclination `inc`
  - validated range for fitting: `0.0 <= inc <= 80.0` degrees
- Total vertical extinction optical depth `tau_max`
  - validated range: `tau_max >= 0.0`

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

- `src/__init__.py`
- `src/stokes_interpolation.py`
- `src/pf_fitting.py`
- `src/stokes_i_fitting.py`
- `data/`
- `notebooks/`

## License

MIT
