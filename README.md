# emergentintensity

Emergent Stokes **I** and **Q** interpolator based on precomputed RT tables
(Kitade & Kataoka 2026).  
Polarization fraction is computed as **Q/I** (no PF tables are used).


## Installation

```bash
git clone https://github.com/astroaki/emergentintensity.git
cd emergentintensity
pip install -e .
```

### Requirements
- Python >= 3.9
- NumPy
- Matplotlib (only for plotting examples/notebooks)

## Quick start

```python
from emergentintensity import emergent_stokes

tau_max = 0.1
omega   = 0.8
inc_deg = 60.0

I, Q = emergent_stokes(tau_max, omega, inc_deg)
print("I/Bnu =", I)
print("Q/Bnu =", Q)
print("Q/I   =", Q/I if I != 0 else 0.0)
```

### Troubleshooting
If you see `INTERP_TABLES is None`, make sure you imported the package from the repository root
(or that the data path is set correctly). The tables are expected under `data/` by default.


## Repository layout
- `src/emergentintensity.py` : main code
- `data/` : RT tables (`taumax*_I_emergent.inp`, `taumax*_Q_emergent.inp`) and `Q_table.inp`
- `notebooks/` : usage & plotting examples

By default, the code searches tables in data/ (repository root).


## Notebooks

Example notebooks in [`notebooks/`](notebooks/):

- [`exampleusage.ipynb`](notebooks/exampleusage.ipynb): basic usage
- [`plot_regimes.ipynb`](notebooks/plot_regimes.ipynb): regime map
- [`plot_fig3.ipynb`](notebooks/plot_fig3.ipynb): draft Fig.3 reproduction
- [`plot_fig4.ipynb`](notebooks/plot_fig4.ipynb): draft Fig.4 reproduction
- [`plot_fig9.ipynb`](notebooks/plot_fig9.ipynb): comparison with analytic formulae
- [`plot_emergent_intensity.ipynb`](notebooks/plot_emergent_intensity.ipynb): plotting helper## Citation

If you use this code or the tables, please cite:
- Kitade & Kataoka (2026, in prep.)

## License
MIT