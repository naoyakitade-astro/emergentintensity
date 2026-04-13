# ============================================================
# Kitade & Kataoka (2026) emergent Stokes calculator
#   - Stokes I and Q interpolator for tau in [1e-2, 15.0]
#   - Using saturated values of I and Q at optically thick side for tau > 15.0
#   - simple analytic formula of Stokes I for tau < 1e-2
#   - Q_table.inp (Eq.C12) is loaded from DATA_DIR and used for tau in [1e-4, 1e-2]
#   - smooth connection for Q in [0.01, 0.03] using Eq.C12 -> RT Hermite bridge
# ============================================================

from pathlib import Path
import re
import numpy as np


# ------------------------------------------------------------
# 0. Global configuration
# ------------------------------------------------------------
MODULE_DIR = Path(__file__).resolve().parent

# Keep the original ``repo_root/data`` default, but resolve paths flexibly so
# the module works both in the public repository layout and when the files are
# placed next to this module.
REPO_ROOT = Path(__file__).resolve().parents[1]

# Base directory that contains ``Q_table.inp`` and/or the folders
# ``StokesI_emergent`` / ``StokesQ_emergent``.
DATA_DIR = str(REPO_ROOT / "data")
DEFAULT_I_DIRNAME = "StokesI_emergent"
DEFAULT_Q_DIRNAME = "StokesQ_emergent"

TABLES = None
INTERP_TABLES = None

# Use Eq.(10) Q-table in this thin regime (requested: 1e-4 to 1e-2)
TAU_EQ10_USE_MIN = 1e-4


# ============================================================
# 1. Reading emergent I/Q tables from split .inp files
# ============================================================

def _unique_paths(paths):
    """
    Return input paths with duplicates removed while preserving order.
    """
    out = []
    seen = set()
    for p in paths:
        rp = Path(p).expanduser().resolve()
        key = str(rp)
        if key not in seen:
            out.append(rp)
            seen.add(key)
    return out


def _candidate_base_dirs(data_dir=None):
    """
    Candidate base directories that may contain the new-format data.

    A valid base directory is expected to contain either
      - Q_table.inp + StokesI_emergent/ + StokesQ_emergent/
    or
      - Q_table.inp + flat files such as I_tau0_1_omega0_1.inp / Q_tau0_1_omega0_1.inp
    """
    candidates = []
    if data_dir is not None:
        candidates.append(Path(data_dir))

    candidates.extend(
        [
            Path(DATA_DIR),
            MODULE_DIR,
            MODULE_DIR.parent,
            REPO_ROOT,
            REPO_ROOT / "data",
        ]
    )
    return _unique_paths(candidates)


def _collect_split_files(directory, kind):
    """
    Collect new-format files in one directory.
    """
    directory = Path(directory)
    files = sorted(directory.glob(f"{kind}_tau*_omega*.inp"))
    if files:
        return files
    return sorted(directory.glob("*.inp"))


def _resolve_emergent_filelists(data_dir=None, idir=None, qdir=None):
    """
    Resolve where the new-format emergent Stokes files live.

    Returns
    -------
    base_dir, files_I, files_Q
    """
    if (idir is None) ^ (qdir is None):
        raise ValueError("Pass both idir and qdir, or neither of them.")

    if idir is not None and qdir is not None:
        i_dir = Path(idir)
        q_dir = Path(qdir)
        files_I = _collect_split_files(i_dir, "I")
        files_Q = _collect_split_files(q_dir, "Q")
        if not files_I:
            raise FileNotFoundError(f"no I inp files found in {i_dir}")
        if not files_Q:
            raise FileNotFoundError(f"no Q inp files found in {q_dir}")
        return i_dir.parent, files_I, files_Q

    for base_dir in _candidate_base_dirs(data_dir):
        i_dir = base_dir / DEFAULT_I_DIRNAME
        q_dir = base_dir / DEFAULT_Q_DIRNAME
        if i_dir.is_dir() and q_dir.is_dir():
            files_I = _collect_split_files(i_dir, "I")
            files_Q = _collect_split_files(q_dir, "Q")
            if files_I and files_Q:
                return base_dir, files_I, files_Q

        # Flat fallback: useful for tests / small samples.
        files_I = sorted(base_dir.glob("I_tau*_omega*.inp"))
        files_Q = sorted(base_dir.glob("Q_tau*_omega*.inp"))
        if files_I and files_Q:
            return base_dir, files_I, files_Q

    searched = "\n  - ".join(str(p) for p in _candidate_base_dirs(data_dir))
    raise FileNotFoundError(
        "Could not find the new-format Stokes data.\n"
        "Looked for either\n"
        f"  - <base>/{DEFAULT_I_DIRNAME} and <base>/{DEFAULT_Q_DIRNAME}, or\n"
        "  - flat files I_tau*_omega*.inp / Q_tau*_omega*.inp\n"
        f"in:\n  - {searched}"
    )



def _resolve_q_table_path(data_dir=None):
    """
    Find Q_table.inp.
    """
    for base_dir in _candidate_base_dirs(data_dir):
        candidate = base_dir / "Q_table.inp"
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Q_table.inp not found in any candidate base directory.")


def load_emergent_file(path):
    """
    Read one new-format split file such as
        I_tau0_1_omega0_1.inp
        Q_tau0_1_omega0_1.inp

    File structure (same parsing style as stokes_i_fitting.py / PF_fitting.py):
        # content = Stokes I emergent intensity only
        # omega = ...
        # tau_max = ...
        # n_mu = ...
        # columns: mu I  (or mu Q)
        <n_mu rows of: mu value>

    Returns a dict:
        {
          "kind": "I" / "Q",
          "omega": float,
          "mu":    np.ndarray (N_mu,),
          "tau_max": float,
          "data": np.ndarray (N_mu,),
        }
    """
    path = Path(path)

    header_lines = []
    data_lines = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                header_lines.append(line.strip())
            elif line.strip():
                data_lines.append(line)

    omega = None
    tau_max = None
    n_mu = None
    kind = None

    for line in header_lines:
        lower = line.lower()
        if line.startswith("# omega ="):
            omega = float(line.split("=", 1)[1].strip())
        elif line.startswith("# tau_max ="):
            tau_max = float(line.split("=", 1)[1].strip())
        elif line.startswith("# n_mu ="):
            n_mu = int(line.split("=", 1)[1].strip())
        elif line.startswith("# content ="):
            if "stokes i" in lower:
                kind = "I"
            elif "stokes q" in lower:
                kind = "Q"
        elif line.startswith("# columns:") and kind is None:
            if lower.endswith(" mu i"):
                kind = "I"
            elif lower.endswith(" mu q"):
                kind = "Q"

    if kind is None:
        stem_lower = path.stem.lower()
        if stem_lower.startswith("i_"):
            kind = "I"
        elif stem_lower.startswith("q_"):
            kind = "Q"

    if kind is None:
        raise ValueError(f"{path}: could not determine whether the file is I or Q")
    if omega is None:
        raise ValueError(f"{path}: omega not found in header")
    if tau_max is None:
        raise ValueError(f"{path}: tau_max not found in header")
    if n_mu is None:
        raise ValueError(f"{path}: n_mu not found in header")
    if not data_lines:
        raise ValueError(f"{path}: no data rows found")

    arr = np.loadtxt(data_lines)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] < 2:
        raise ValueError(f"{path}: expected at least two columns (mu, value)")

    mu = np.asarray(arr[:, 0], dtype=np.float64)
    data = np.asarray(arr[:, 1], dtype=np.float64)

    if mu.size != n_mu:
        raise ValueError(f"{path}: header n_mu and actual row count do not match")

    return {
        "kind": kind,
        "omega": float(omega),
        "mu": mu,
        "tau_max": float(tau_max),
        "data": data,
    }


def _build_record_map(filelist, expected_kind=None):
    """
    Build a dict keyed by (tau_max, omega) from split emergent files.
    """
    records = {}
    for filepath in filelist:
        rec = load_emergent_file(filepath)
        if expected_kind is not None and rec["kind"] != expected_kind:
            raise ValueError(
                f"Unexpected kind in {filepath}: got {rec['kind']}, expected {expected_kind}"
            )

        key = (float(rec["tau_max"]), float(rec["omega"]))
        if key in records:
            raise ValueError(
                f"Duplicate {rec['kind']} file for tau={key[0]}, omega={key[1]}: {filepath}"
            )
        records[key] = rec
    return records


def collect_emergent_tables(data_dir=None, idir=None, qdir=None):
    """
    Scan the new-format emergent files and rebuild the old interpolation-ready
    table structure:

        {
          tau1: {"I": rec_I, "Q": rec_Q},
          tau2: {"I": ...,  "Q": ...},
          ...
        }

    where each ``rec_I`` / ``rec_Q`` matches the shape expected by the rest of
    this module:
        {
          "kind": "I" / "Q",
          "omega": np.ndarray (N_omega,),
          "mu":    np.ndarray (N_mu,),
          "tau_max": float,
          "data": np.ndarray (N_omega, N_mu),
        }

    Notes
    -----
    - ``omega=0`` may be absent from the Q files. In that case a synthetic
      zero-Q row is added from the corresponding I mu-grid.
    - Apart from this reconstruction step, the downstream interpolation code is
      kept unchanged.
    """
    _, filelist_I, filelist_Q = _resolve_emergent_filelists(
        data_dir=data_dir,
        idir=idir,
        qdir=qdir,
    )

    records_I = _build_record_map(filelist_I, expected_kind="I")
    records_Q = _build_record_map(filelist_Q, expected_kind="Q")

    # If Q(omega=0) is omitted, synthesize it as identically zero so the rest
    # of the code can keep the same (tau, omega) table structure.
    for key, rec_I in list(records_I.items()):
        tau, omega = key
        if np.isclose(omega, 0.0) and key not in records_Q:
            records_Q[key] = {
                "kind": "Q",
                "omega": float(omega),
                "mu": np.asarray(rec_I["mu"], dtype=np.float64),
                "tau_max": float(tau),
                "data": np.zeros_like(rec_I["data"], dtype=np.float64),
            }

    keys_I = set(records_I.keys())
    keys_Q = set(records_Q.keys())

    missing_in_Q = sorted(
        key for key in keys_I if (key not in keys_Q) and (not np.isclose(key[1], 0.0))
    )
    missing_in_I = sorted(key for key in keys_Q if key not in keys_I)
    if missing_in_Q or missing_in_I:
        raise ValueError(
            "Stokes I and Stokes Q grids do not match.\n"
            f"Missing in Q: {missing_in_Q}\n"
            f"Missing in I: {missing_in_I}"
        )


    tau_list = sorted({tau for tau, _ in keys_I})
    tables = {}

    for tau in tau_list:
        omega_list = sorted(omega for t, omega in keys_I if t == tau)
        mu_ref = None
        I_rows = []
        Q_rows = []

        for omega in omega_list:
            rec_I = records_I[(tau, omega)]
            rec_Q = records_Q[(tau, omega)]

            if not np.allclose(rec_I["mu"], rec_Q["mu"]):
                raise ValueError(
                    f"mu grid mismatch between I and Q at tau={tau}, omega={omega}"
                )

            if mu_ref is None:
                mu_ref = np.asarray(rec_I["mu"], dtype=np.float64)
            elif not np.allclose(mu_ref, rec_I["mu"]):
                raise ValueError(
                    f"mu grid mismatch across omega files at tau={tau}"
                )

            I_rows.append(np.asarray(rec_I["data"], dtype=np.float64))
            Q_rows.append(np.asarray(rec_Q["data"], dtype=np.float64))

        data_I = np.vstack(I_rows)
        data_Q = np.vstack(Q_rows)
        omega_arr = np.asarray(omega_list, dtype=np.float64)

        tables[float(tau)] = {
            "I": {
                "kind": "I",
                "omega": omega_arr,
                "mu": np.asarray(mu_ref, dtype=np.float64),
                "tau_max": float(tau),
                "data": data_I,
            },
            "Q": {
                "kind": "Q",
                "omega": omega_arr,
                "mu": np.asarray(mu_ref, dtype=np.float64),
                "tau_max": float(tau),
                "data": data_Q,
            },
        }

    return tables

# ============================================================
# 2. Build interpolation-friendly structure
# ============================================================

def build_interp_tables(tables):
    """
    Convert collected tables into a structure for interpolation.

    Returns:
        {
          "tau_grid":   np.ndarray (N_tau,),
          "omega_grid": np.ndarray (N_omega,),
          "per_tau": [
              {
                "mu_grid": np.ndarray (N_mu_this_tau,),
                "I":  np.ndarray (N_omega, N_mu_this_tau),
                "Q":  np.ndarray (N_omega, N_mu_this_tau),
              },
              ...
          ]
        }
    """
    tau_list = sorted(tables.keys())
    tau_grid = np.array(tau_list, dtype=np.float64)

    ref = tables[tau_list[0]]["I"]
    omega_ref = np.asarray(ref["omega"], dtype=np.float64)

    per_tau = []
    for tau in tau_list:
        bucket = tables[tau]
        rec_I = bucket["I"]
        rec_Q = bucket["Q"]

        if not (np.allclose(rec_I["omega"], omega_ref) and np.allclose(rec_Q["omega"], omega_ref)):
            raise ValueError(f"omega grid mismatch at tau={tau}")

        if not np.allclose(rec_I["mu"], rec_Q["mu"]):
            raise ValueError(f"mu grid mismatch between I and Q at tau={tau}")

        Idata = np.asarray(rec_I["data"], dtype=np.float32)
        Qdata = np.asarray(rec_Q["data"], dtype=np.float32)

        if Idata.shape != Qdata.shape:
            raise ValueError(f"data shape mismatch at tau={tau}: I{Idata.shape}, Q{Qdata.shape}")

        per_tau.append(
            {
                "mu_grid": np.asarray(rec_I["mu"], dtype=np.float32),
                "I": Idata,
                "Q": Qdata,
            }
        )

    return {
        "tau_grid": tau_grid,
        "omega_grid": omega_ref.astype(np.float32),
        "per_tau": per_tau,
    }


def setup_tables(data_dir=DATA_DIR, idir=None, qdir=None):
    """
    Read new-format Stokes I/Q files and build interpolation tables.

    Parameters
    ----------
    data_dir : str or Path, optional
        Base directory that contains ``Q_table.inp`` and either
        ``StokesI_emergent`` / ``StokesQ_emergent`` or flat ``I_tau...`` /
        ``Q_tau...`` files.
    idir, qdir : str or Path, optional
        Explicit I/Q directories. When given, these override automatic path
        discovery.
    """
    global TABLES, INTERP_TABLES
    TABLES = collect_emergent_tables(data_dir=data_dir, idir=idir, qdir=qdir)
    INTERP_TABLES = build_interp_tables(TABLES)
    return TABLES, INTERP_TABLES


# ============================================================
# 3. Low-level interpolation in omega, mu
# ============================================================

def _interp_along_mu(mu, mu_grid, data_omega_mu):
    """
    Linear interpolation along mu for each omega.
    Returns shape (N_omega,)
    """
    mu_grid = np.asarray(mu_grid)
    data = np.asarray(data_omega_mu)
    mu = float(mu)

    mu_c = float(np.clip(mu, mu_grid[0], mu_grid[-1]))

    idx_hi = int(np.searchsorted(mu_grid, mu_c, side="right"))
    idx_hi = int(np.clip(idx_hi, 1, mu_grid.shape[0] - 1))
    idx_lo = idx_hi - 1

    mu0 = float(mu_grid[idx_lo])
    mu1 = float(mu_grid[idx_hi])
    t = (mu_c - mu0) / (mu1 - mu0)

    val0 = data[:, idx_lo]
    val1 = data[:, idx_hi]
    return (1.0 - t) * val0 + t * val1


def _interp1d_omega(omega, omega_grid, values):
    """
    1D linear interpolation along omega.
    """
    omega_grid = np.asarray(omega_grid)
    values = np.asarray(values)
    omega = float(omega)

    oc = float(np.clip(omega, omega_grid[0], omega_grid[-1]))

    idx_hi = int(np.searchsorted(omega_grid, oc, side="right"))
    idx_hi = int(np.clip(idx_hi, 1, omega_grid.shape[0] - 1))
    idx_lo = idx_hi - 1

    w0 = float(omega_grid[idx_lo])
    w1 = float(omega_grid[idx_hi])
    t = (oc - w0) / (w1 - w0)

    y0 = float(values[idx_lo])
    y1 = float(values[idx_hi])
    return float((1.0 - t) * y0 + t * y1)


def _interp2d(omega, mu, omega_grid, mu_grid, data_omega_mu):
    """
    2D interpolation in (omega, mu) using linear interpolation in both directions.
    Returns scalar.
    """
    vals_vs_omega = _interp_along_mu(mu, mu_grid, data_omega_mu)  # shape (N_omega,)
    return _interp1d_omega(omega, omega_grid, vals_vs_omega)


# ============================================================
# 4. Smooth interpolation along tau (cubic Hermite)
# ============================================================

def interp_stokes_scalar_smooth(tau_max, omega, mu, kind="I"):
    """
    Low-level interpolator:
      (tau_max, omega, mu) -> kind ("I" or "Q")
    """
    if INTERP_TABLES is None:
        raise RuntimeError("INTERP_TABLES is None. Call setup_tables() first.")

    tau_max = float(tau_max)
    omega = float(omega)
    mu = float(mu)

    # almost pure absorption
    if abs(omega) < 1e-6:
        mu_eff = max(mu, 1e-3)
        if kind == "I":
            return float(1.0 - np.exp(-tau_max / mu_eff))
        elif kind == "Q":
            return 0.0
        else:
            raise ValueError(f"unknown kind: {kind}")

    tau_grid = INTERP_TABLES["tau_grid"]
    omega_grid = INTERP_TABLES["omega_grid"]
    per_tau = INTERP_TABLES["per_tau"]

    tau_min = float(tau_grid[0])
    tau_max_grid = float(tau_grid[-1])
    tau_c = float(np.clip(tau_max, tau_min, tau_max_grid))

    N = len(tau_grid)
    idx_hi = int(np.searchsorted(tau_grid, tau_c, side="right"))
    idx_hi = max(1, min(idx_hi, N - 1))
    idx_lo = idx_hi - 1

    # Left edge interval
    if idx_lo == 0:
        k0, k1, k2 = 0, 1, 2
        tau0 = float(tau_grid[k0])
        tau1 = float(tau_grid[k1])
        tau2 = float(tau_grid[k2])

        rec0 = per_tau[k0]
        rec1 = per_tau[k1]
        rec2 = per_tau[k2]

        y0 = _interp2d(omega, mu, omega_grid, rec0["mu_grid"], rec0[kind])
        y1 = _interp2d(omega, mu, omega_grid, rec1["mu_grid"], rec1[kind])
        y2 = _interp2d(omega, mu, omega_grid, rec2["mu_grid"], rec2[kind])

        d01 = (y1 - y0) / (tau1 - tau0)
        d12 = (y2 - y1) / (tau2 - tau1)

        m0 = d01
        m1 = 0.5 * (d01 + d12)

        s = (tau_c - tau0) / (tau1 - tau0)
        h00 =  2.0*s**3 - 3.0*s**2 + 1.0
        h10 =        s**3 - 2.0*s**2 + s
        h01 = -2.0*s**3 + 3.0*s**2
        h11 =        s**3 -       s**2
        dt = tau1 - tau0

        return float(h00*y0 + h10*dt*m0 + h01*y1 + h11*dt*m1)

    # Right edge interval
    if idx_hi == N - 1:
        k0, k1, k2 = N - 3, N - 2, N - 1
        tau0 = float(tau_grid[k0])
        tau1 = float(tau_grid[k1])
        tau2 = float(tau_grid[k2])

        rec0 = per_tau[k0]
        rec1 = per_tau[k1]
        rec2 = per_tau[k2]

        y0 = _interp2d(omega, mu, omega_grid, rec0["mu_grid"], rec0[kind])
        y1 = _interp2d(omega, mu, omega_grid, rec1["mu_grid"], rec1[kind])
        y2 = _interp2d(omega, mu, omega_grid, rec2["mu_grid"], rec2[kind])

        d01 = (y1 - y0) / (tau1 - tau0)
        d12 = (y2 - y1) / (tau2 - tau1)

        m1 = 0.5 * (d01 + d12)
        m2 = d12

        s = (tau_c - tau1) / (tau2 - tau1)
        h00 =  2.0*s**3 - 3.0*s**2 + 1.0
        h10 =        s**3 - 2.0*s**2 + s
        h01 = -2.0*s**3 + 3.0*s**2
        h11 =        s**3 -       s**2
        dt = tau2 - tau1

        return float(h00*y1 + h10*dt*m1 + h01*y2 + h11*dt*m2)

    # Interior intervals
    k0 = idx_lo - 1
    k1 = idx_lo
    k2 = idx_hi
    k3 = idx_hi + 1

    tau0 = float(tau_grid[k0])
    tau1 = float(tau_grid[k1])
    tau2 = float(tau_grid[k2])
    tau3 = float(tau_grid[k3])

    rec0 = per_tau[k0]
    rec1 = per_tau[k1]
    rec2 = per_tau[k2]
    rec3 = per_tau[k3]

    y0 = _interp2d(omega, mu, omega_grid, rec0["mu_grid"], rec0[kind])
    y1 = _interp2d(omega, mu, omega_grid, rec1["mu_grid"], rec1[kind])
    y2 = _interp2d(omega, mu, omega_grid, rec2["mu_grid"], rec2[kind])
    y3 = _interp2d(omega, mu, omega_grid, rec3["mu_grid"], rec3[kind])

    d10 = (y1 - y0) / (tau1 - tau0)
    d21 = (y2 - y1) / (tau2 - tau1)
    d32 = (y3 - y2) / (tau3 - tau2)

    m1 = 0.5 * (d21 + d10)
    m2 = 0.5 * (d32 + d21)

    s = (tau_c - tau1) / (tau2 - tau1)
    h00 =  2.0*s**3 - 3.0*s**2 + 1.0
    h10 =        s**3 - 2.0*s**2 + s
    h01 = -2.0*s**3 + 3.0*s**2
    h11 =        s**3 -       s**2
    dt = tau2 - tau1

    return float(h00*y1 + h10*dt*m1 + h01*y2 + h11*dt*m2)


# ============================================================
# 5. Public pure-interpolation API (no thin/thick patches)
# ============================================================

def interpolate_stokes(tau_max, omega, inc_deg):
    """
    Pure interpolation from RT tables:
        (tau_max, omega, inc_deg [deg]) -> (I, Q)

    No optically thin / thick patches are applied.
    """
    inc_rad = np.deg2rad(float(inc_deg))
    mu = float(np.cos(inc_rad))

    I = interp_stokes_scalar_smooth(tau_max, omega, mu, kind="I")
    Q = interp_stokes_scalar_smooth(tau_max, omega, mu, kind="Q")
    return float(I), float(Q)


# ============================================================
# 6. Optically thin approximation for I and fallback Q
# ============================================================

def analytic_thin_I_scalar(tau_max, omega, inc_deg):
    """
    Optically thin analytic solution for emergent intensity:

        I/B = 1 - exp( -tau_abs / mu )
        tau_abs = tau_max * (1 - omega)
    """
    tau_max = float(tau_max)
    omega = float(omega)
    inc_deg = float(inc_deg)

    mu = float(np.cos(np.deg2rad(inc_deg)))
    mu = max(mu, 1e-10)

    tau_abs = tau_max * (1.0 - omega)
    tau_los = tau_abs / mu
    return float(1.0 - np.exp(-tau_los))


def Q_thin_scalar(tau_max, omega, inc_deg):
    """
    Optically thin fallback approximation for Stokes Q:
        Q_thin ∝ (1 - mu^2)/mu * omega*(1-omega) * tau_max^2
    The proportionality constant is matched to RT at tau_ref = tau_min_table.
    """
    if INTERP_TABLES is None:
        raise RuntimeError("INTERP_TABLES is None. Call setup_tables() first.")

    tau_max = float(tau_max)
    omega = float(omega)
    inc_deg = float(inc_deg)

    if tau_max <= 0.0 or omega <= 0.0 or omega >= 1.0:
        return 0.0

    tau_grid = INTERP_TABLES["tau_grid"]
    tau_ref = float(tau_grid[0])

    mu = float(np.cos(np.deg2rad(inc_deg)))
    mu = float(np.clip(mu, 1e-10, 1.0))

    _, Q_ref = interpolate_stokes(tau_ref, omega, inc_deg)

    base_ref = ((1.0 - mu**2) / mu) * omega * (1.0 - omega) * (tau_ref**2)
    eps = 1e-30
    if abs(base_ref) < eps:
        return float(Q_ref * (tau_max / tau_ref)**2)

    c_Q = Q_ref / base_ref
    base_t = ((1.0 - mu**2) / mu) * omega * (1.0 - omega) * (tau_max**2)
    return float(c_Q * base_t)


# ============================================================
# Extra thin-regime Q table from draft Eq. (C12)
# (file: DATA_DIR/Q_table.inp)
# ============================================================

Q_THIN_TABLE = None  # will be set by load_Q_thin_table()


def load_Q_thin_table(path):
    """
    Load the small-tau Stokes Q table computed from draft Eq. (C12).

    File format (Q_table.inp):
        # Q table shape: n_mu n_omega n_tau
        n_mu n_omega n_tau
        # mu
        <n_mu values>
        # omega
        <n_omega values>
        # tau
        <n_tau values>
        # data[mu_index, omega_index, tau_index]
        <n_mu * n_omega lines of Q/B_nu>
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Q thin table not found: {path}")

    with path.open() as f:
        _ = f.readline()  # comment line
        shape_line = f.readline().strip()
        n_mu, n_omega, n_tau = map(int, shape_line.split())

        line = f.readline().strip()
        assert line.startswith("# mu")
        mu_vals = np.fromstring(f.readline(), sep=" ")
        assert mu_vals.size == n_mu

        line = f.readline().strip()
        assert line.startswith("# omega")
        omega_vals = np.fromstring(f.readline(), sep=" ")
        assert omega_vals.size == n_omega

        line = f.readline().strip()
        assert line.startswith("# tau")
        tau_vals = np.fromstring(f.readline(), sep=" ")
        assert tau_vals.size == n_tau

        _ = f.readline()
        _ = f.readline()
        _ = f.readline()

        data = np.empty((n_mu, n_omega, n_tau), dtype=float)
        for imu in range(n_mu):
            for iw in range(n_omega):
                row = np.fromstring(f.readline(), sep=" ")
                if row.size != n_tau:
                    raise ValueError(
                        f"Unexpected number of tau values at (imu={imu}, iw={iw}): "
                        f"got {row.size}, expected {n_tau}"
                    )
                data[imu, iw, :] = row

    return {
        "mu_grid": mu_vals,
        "omega_grid": omega_vals,
        "tau_grid": tau_vals,
        "data": data,
    }


# load the table once (from the resolved base directory)
try:
    _Q_THIN_TABLE_PATH = _resolve_q_table_path(DATA_DIR)
    Q_THIN_TABLE = load_Q_thin_table(_Q_THIN_TABLE_PATH)
    # print(f"Loaded Q_THIN_TABLE from {_Q_THIN_TABLE_PATH}")
except FileNotFoundError:
    print("Q_table.inp not found; Q_THIN_TABLE will remain None.")


def Q_eqC12_table_scalar(tau_max, omega, inc_deg):
    """
    Q/B_nu from the small-tau table computed using draft Eq. (C12).
    Trilinear interpolation in (mu, omega, tau).
    """
    if Q_THIN_TABLE is None:
        raise RuntimeError("Q_THIN_TABLE is None. Call load_Q_thin_table() first.")

    tau_max = float(tau_max)
    omega = float(omega)
    inc_deg = float(inc_deg)

    mu = float(np.cos(np.deg2rad(inc_deg)))
    mu = float(np.clip(mu, 0.0, 1.0))

    mu_grid = Q_THIN_TABLE["mu_grid"]
    omega_grid = Q_THIN_TABLE["omega_grid"]
    tau_grid = Q_THIN_TABLE["tau_grid"]
    data = Q_THIN_TABLE["data"]  # (n_mu, n_omega, n_tau)

    tau_min_thin = float(tau_grid[0])
    tau_max_thin = float(tau_grid[-1])
    tau_c = float(np.clip(tau_max, tau_min_thin, tau_max_thin))

    # tau indices
    if tau_c <= tau_grid[0]:
        it_lo = it_hi = 0
        tt = 0.0
    elif tau_c >= tau_grid[-1]:
        it_lo = it_hi = len(tau_grid) - 1
        tt = 0.0
    else:
        it_hi = int(np.searchsorted(tau_grid, tau_c, side="right"))
        it_lo = it_hi - 1
        t0 = float(tau_grid[it_lo])
        t1 = float(tau_grid[it_hi])
        tt = (tau_c - t0) / (t1 - t0)

    # omega indices
    if omega <= omega_grid[0]:
        iw_lo = iw_hi = 0
        to = 0.0
    elif omega >= omega_grid[-1]:
        iw_lo = iw_hi = len(omega_grid) - 1
        to = 0.0
    else:
        iw_hi = int(np.searchsorted(omega_grid, omega, side="right"))
        iw_lo = iw_hi - 1
        o0 = float(omega_grid[iw_lo])
        o1 = float(omega_grid[iw_hi])
        to = (omega - o0) / (o1 - o0)

    # mu indices
    if mu <= mu_grid[0]:
        im_lo = im_hi = 0
        tm = 0.0
    elif mu >= mu_grid[-1]:
        im_lo = im_hi = len(mu_grid) - 1
        tm = 0.0
    else:
        im_hi = int(np.searchsorted(mu_grid, mu, side="right"))
        im_lo = im_hi - 1
        m0 = float(mu_grid[im_lo])
        m1 = float(mu_grid[im_hi])
        tm = (mu - m0) / (m1 - m0)

    def lerp(a, b, t):
        return (1.0 - t) * a + t * b

    def D(im, iw, it):
        return float(data[im, iw, it])

    C000 = D(im_lo, iw_lo, it_lo)
    C001 = D(im_lo, iw_lo, it_hi)
    C010 = D(im_lo, iw_hi, it_lo)
    C011 = D(im_lo, iw_hi, it_hi)
    C100 = D(im_hi, iw_lo, it_lo)
    C101 = D(im_hi, iw_lo, it_hi)
    C110 = D(im_hi, iw_hi, it_lo)
    C111 = D(im_hi, iw_hi, it_hi)

    C00 = lerp(C000, C001, tt)
    C01 = lerp(C010, C011, tt)
    C10 = lerp(C100, C101, tt)
    C11 = lerp(C110, C111, tt)

    C0 = lerp(C00, C01, to)
    C1 = lerp(C10, C11, to)

    C = lerp(C0, C1, tm)
    return float(C)


def Q_C8_scalar(tau_max, omega, inc_deg):
    """
    Extremely thin limit for Stokes Q using Eq. (C8)-like form:

        Q/B_nu ∝ (1 - mu^2)/mu * omega * (1 - omega) * tau_max^2

    The proportionality constant c_q is determined to match Eq.(C12) table
    at the smallest tau in Q_THIN_TABLE.
    """
    if Q_THIN_TABLE is None:
        return Q_thin_scalar(tau_max, omega, inc_deg)

    tau_max = float(tau_max)
    omega = float(omega)
    inc_deg = float(inc_deg)

    if tau_max <= 0.0 or omega <= 0.0 or omega >= 1.0:
        return 0.0

    tau_grid_thin = Q_THIN_TABLE["tau_grid"]
    tau_ref = float(tau_grid_thin[0])

    mu = float(np.cos(np.deg2rad(inc_deg)))
    mu = float(np.clip(mu, 1e-10, 1.0))

    Q_ref = Q_eqC12_table_scalar(tau_ref, omega, inc_deg)

    base_ref = ((1.0 - mu**2) / mu) * omega * (1.0 - omega) * (tau_ref**2)
    eps = 1e-30
    if abs(base_ref) < eps:
        return float(Q_ref * (tau_max / tau_ref)**2)

    c_q = Q_ref / base_ref
    base_t = ((1.0 - mu**2) / mu) * omega * (1.0 - omega) * (tau_max**2)
    return float(c_q * base_t)


def _deriv_at_x1_quadratic(x0, x1, x2, y0, y1, y2):
    """
    derivative at x1 of quadratic through (x0,y0),(x1,y1),(x2,y2)
    """
    return (
        y0*(x1-x2)/((x0-x1)*(x0-x2))
        + y1*(2*x1-x0-x2)/((x1-x0)*(x1-x2))
        + y2*(x1-x0)/((x2-x0)*(x2-x1))
    )


def Q_bridge_small_tau(tau_max, omega, inc_deg):
    """
    Smoothly connect Q from Eq.(C12) thin table to RT tables
    in the first RT interval [tau0, tau1] using cubic Hermite.

    This requires Q_THIN_TABLE (Eq10).

    IMPORTANT (requested behavior):
      - use Eq.(C12) table in tau range [TAU_EQ10_USE_MIN, tau0(=tau_min_table)]
      - bridge in [tau0, tau1] with Q0 evaluated at tau0 (NOT at thin-table last point)
      - compute slope m0 from Eq.(C12) using the point just below tau0
    """
    if (INTERP_TABLES is None) or (Q_THIN_TABLE is None):
        raise RuntimeError("Both INTERP_TABLES and Q_THIN_TABLE are required.")

    tau_max = float(tau_max)
    omega = float(omega)
    inc_deg = float(inc_deg)

    tau_grid = INTERP_TABLES["tau_grid"]
    tau0 = float(tau_grid[0])  # e.g. 0.01
    tau1 = float(tau_grid[1])  # e.g. 0.03
    tau2 = float(tau_grid[2])  # e.g. 0.05

    if not (tau0 <= tau_max <= tau1):
        raise ValueError("Q_bridge_small_tau is only valid for tau in [tau0, tau1].")

    # ---- Left endpoint from Eq.(C12) side: value at tau0 ----
    Q0 = float(Q_eqC12_table_scalar(tau0, omega, inc_deg))

    # ---- Left slope m0 from Eq.(C12): use tau just below tau0 (backward diff) ----
    tg = np.asarray(Q_THIN_TABLE["tau_grid"], dtype=float)
    k = int(np.searchsorted(tg, tau0, side="right") - 1)  # tg[k] <= tau0

    if k <= 0:
        # If no previous point exists, use first two points
        tA = float(tg[0])
        tB = float(tg[1])
        QA = float(Q_eqC12_table_scalar(tA, omega, inc_deg))
        QB = float(Q_eqC12_table_scalar(tB, omega, inc_deg))
        m0 = (QB - QA) / (tB - tA)
    else:
        t_prev = float(tg[k-1])
        Q_prev = float(Q_eqC12_table_scalar(t_prev, omega, inc_deg))
        m0 = (Q0 - Q_prev) / (tau0 - t_prev)

    # ---- Right endpoint from RT side ----
    _, Q1 = interpolate_stokes(tau1, omega, inc_deg)

    # RT-side derivative at tau1 (quadratic fit using tau0,tau1,tau2)
    _, Q0_rt = interpolate_stokes(tau0, omega, inc_deg)
    _, Q2_rt = interpolate_stokes(tau2, omega, inc_deg)
    m1 = _deriv_at_x1_quadratic(tau0, tau1, tau2, Q0_rt, Q1, Q2_rt)

    # ---- Hermite in [tau0, tau1] ----
    s = (tau_max - tau0) / (tau1 - tau0)
    h00 =  2.0*s**3 - 3.0*s**2 + 1.0
    h10 =        s**3 - 2.0*s**2 + s
    h01 = -2.0*s**3 + 3.0*s**2
    h11 =        s**3 -       s**2
    dt = tau1 - tau0

    Q_bridge = h00*Q0 + h10*dt*m0 + h01*Q1 + h11*dt*m1
    return float(Q_bridge)


def Q_bridge_tau0_tau1_fallback(tau_max, omega, inc_deg):
    """
    Fallback bridge when Q_THIN_TABLE is None.
    Smoothly connect thin model (Q_thin_scalar) to RT in [tau0, tau1].
    """
    if INTERP_TABLES is None:
        raise RuntimeError("INTERP_TABLES is None. Call setup_tables() first.")

    tau_max = float(tau_max)
    omega = float(omega)
    inc_deg = float(inc_deg)

    tau_grid = INTERP_TABLES["tau_grid"]
    tau0 = float(tau_grid[0])
    tau1 = float(tau_grid[1])
    tau2 = float(tau_grid[2])

    if tau_max <= tau0:
        return float(Q_thin_scalar(tau_max, omega, inc_deg))
    if tau_max >= tau1:
        _, Q = interpolate_stokes(tau_max, omega, inc_deg)
        return float(Q)

    # left: thin value and slope (assume ~tau^2)
    Q0 = float(Q_thin_scalar(tau0, omega, inc_deg))
    m0 = 0.0 if tau0 == 0.0 else (2.0 * Q0 / tau0)

    # right: RT value and slope
    _, Q1 = interpolate_stokes(tau1, omega, inc_deg)
    _, Q0_rt = interpolate_stokes(tau0, omega, inc_deg)
    _, Q2_rt = interpolate_stokes(tau2, omega, inc_deg)
    m1 = _deriv_at_x1_quadratic(tau0, tau1, tau2, Q0_rt, Q1, Q2_rt)

    # Hermite
    s = (tau_max - tau0) / (tau1 - tau0)
    h00 =  2.0*s**3 - 3.0*s**2 + 1.0
    h10 =        s**3 - 2.0*s**2 + s
    h01 = -2.0*s**3 + 3.0*s**2
    h11 =        s**3 -       s**2
    dt = tau1 - tau0

    return float(h00*Q0 + h10*dt*m0 + h01*Q1 + h11*dt*m1)


# ============================================================
# 7. High-level emergent API with thin/thick patches
# ============================================================

def emergent_stokes(tau_max, omega, inc_deg):
    """
    High-level API:
        (tau_max, omega, inc_deg [deg]) -> (I, Q)

    This function applies:
      - omega=0 : pure absorption I, Q=0
      - tau below RT min : thin patches
          I: analytic_thin_I_scalar
          Q: Eq.(10) table in [TAU_EQ10_USE_MIN, tau0], B4 below that (if table exists),
             otherwise fallback Q_thin_scalar
      - first RT interval [tau0, tau1] : Q bridged smoothly (Eq10->RT, or fallback->RT)
      - tau above RT max : saturation at tau_max_table (RT)
      - otherwise : pure RT interpolation
    """
    if INTERP_TABLES is None:
        raise RuntimeError("INTERP_TABLES is None. Call setup_tables() first.")

    tau_max = float(tau_max)
    omega = float(omega)
    inc_deg = float(inc_deg)

    if tau_max <= 0.0:
        return 0.0, 0.0

    tau_grid = INTERP_TABLES["tau_grid"]
    tau_min_table = float(tau_grid[0])
    tau_max_table = float(tau_grid[-1])

    # first RT interval [tau0, tau1]
    tau0 = float(tau_grid[0])  # e.g. 0.01
    tau1 = float(tau_grid[1])  # e.g. 0.03

    # omega = 0: pure absorption solution
    if omega == 0.0:
        I = analytic_thin_I_scalar(tau_max, omega, inc_deg)
        Q = 0.0
        return float(I), float(Q)

    # 1) optically thin region below RT tables
    if tau_max < tau_min_table:
        I = analytic_thin_I_scalar(tau_max, omega, inc_deg)

        if Q_THIN_TABLE is not None:
            tau_grid_thin = np.asarray(Q_THIN_TABLE["tau_grid"], dtype=float)
            tau_min_thin = float(tau_grid_thin[0])
            tau_max_thin = float(tau_grid_thin[-1])

            # requested: prefer Eq.(C12) table for tau in [1e-4, 1e-2]
            if tau_max < TAU_EQ10_USE_MIN:
                # ultra-thin: extend using B4 form matched to Eq10 minimum
                Q = Q_C8_scalar(tau_max, omega, inc_deg)
            else:
                # use Eq10 table (clip to its coverage)
                tau_for_eq10 = float(np.clip(tau_max, tau_min_thin, tau_max_thin))
                Q = Q_eqC12_table_scalar(tau_for_eq10, omega, inc_deg)
        else:
            # no Eq10 table -> fallback
            Q = Q_thin_scalar(tau_max, omega, inc_deg)

        return float(I), float(Q)

    # 2) first RT interval: bridge Q smoothly (regardless of Eq10 availability)
    if tau0 <= tau_max <= tau1:
        I, _ = interpolate_stokes(tau_max, omega, inc_deg)

        if Q_THIN_TABLE is not None:
            Q = Q_bridge_small_tau(tau_max, omega, inc_deg)
        else:
            Q = Q_bridge_tau0_tau1_fallback(tau_max, omega, inc_deg)

        return float(I), float(Q)

    # 3) very thick: saturate at tau_max_table
    if tau_max > tau_max_table:
        I_sat, Q_sat = interpolate_stokes(tau_max_table, omega, inc_deg)
        return float(I_sat), float(Q_sat)

    # 4) intermediate: pure RT interpolation
    I, Q = interpolate_stokes(tau_max, omega, inc_deg)
    return float(I), float(Q)


def emergent_polarization(tau_max, omega, inc_deg, mode="Q_over_I"):
    """
    Convenience wrapper computed from (I, Q) without any PF table.

    mode:
      - "Q_over_I": return Q/I
      - "abs_Q_over_I": return |Q|/I
      - "signed_abs": return sign(Q) * |Q|/I (same as Q/I, but explicit)

    Note: This function is optional; it does not reintroduce PF tables.
    """
    I, Q = emergent_stokes(tau_max, omega, inc_deg)
    I = float(I)
    Q = float(Q)
    if I == 0.0:
        return 0.0

    if mode == "Q_over_I":
        return float(Q / I)
    elif mode == "abs_Q_over_I":
        return float(abs(Q) / I)
    elif mode == "signed_abs":
        return float(np.sign(Q) * (abs(Q) / I))
    else:
        raise ValueError(f"unknown mode: {mode}")


# ============================================================
# 8. Initialize tables and simple sanity check
# ============================================================

TABLES, INTERP_TABLES = setup_tables(DATA_DIR)
'''
print("Loaded tau grid and available kinds (loaded from files):")
for tau in sorted(TABLES.keys()):
    kinds = [k for k in ["I", "Q"] if k in TABLES[tau]]
    print(f"  tau = {tau:g}  kinds: {kinds}")

print("\nSanity check for tau grids:")
print("RT tau_grid[:3] =", INTERP_TABLES["tau_grid"][:3])
print("Q_THIN_TABLE loaded? ->", Q_THIN_TABLE is not None)
if Q_THIN_TABLE is not None:
    print("Eq10 tau_grid[0:3]  =", Q_THIN_TABLE["tau_grid"][:3])
    print("Eq10 tau_grid[-3:] =", Q_THIN_TABLE["tau_grid"][-3:])
    print("TAU_EQ10_USE_MIN =", TAU_EQ10_USE_MIN)
'''


