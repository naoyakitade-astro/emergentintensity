# emergentintensity

Emergent Stokes **I** and **Q** interpolator based on precomputed RT tables
(Kitade & Kataoka 2026).  
Polarization fraction is computed as **Q/I** (no PF tables are used).

## Repository layout
- `src/emergentintensity.py` : main code
- `data/` : RT tables (`taumax*_I_emergent.inp`, `taumax*_Q_emergent.inp`) and `Q_table.inp`
- `notebooks/` : usage & plotting examples

## Install (editable)
```bash
pip install -e .