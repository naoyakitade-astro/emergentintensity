# emergentintensity

Emergent Stokes **I** and **Q** interpolator based on precomputed RT tables
(Kitade & Kataoka 2026).  
Polarization fraction is computed as **Q/I** (no PF tables are used).


## Installation

```bash
git clone https://github.com/astroaki/emergentintensity.git
cd emergentintensity
pip install -e .

## Quick start

from emergentintensity import emergent_stokes

tau_max = 0.1
omega   = 0.8
inc_deg = 60.0

I, Q = emergent_stokes(tau_max, omega, inc_deg)
print("I/Bnu =", I)
print("Q/Bnu =", Q)
print("Q/I   =", Q/I if I != 0 else 0.0)


## Repository layout
- `src/emergentintensity.py` : main code
- `data/` : RT tables (`taumax*_I_emergent.inp`, `taumax*_Q_emergent.inp`) and `Q_table.inp`
- `notebooks/` : usage & plotting examples

By default, the code searches tables in data/ (repository root).


## Notebooks

Example notebooks are provided in notebooks/:
	•	plot_regimes.ipynb: regime map
	•	plot_fig3.ipynb, plot_fig4.ipynb: draft figure reproductions
	•	plot_fig9.ipynb: comparison with analytic formulae
	•	exampleusage.ipynb: basic usage

## Citation

If you use this code or the tables, please cite:
	•	Kitade & Kataoka (2026, in prep.)