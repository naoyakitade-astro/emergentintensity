# ============================================================
# Kitade & Kataoka (2026) emergent Stokes calculator
#   - Stokes I and Q interpolator for tau in [1e-2, 15.0]
#   - Using saturated values of I and Q at optically thick side for tau > 15.0
#   - simple analytic formula of Stokes I for tau < 1e-2
#   - Q_table.inp (Eq.C12) is loaded from DATA_DIR and used for tau in [1e-4, 1e-2]
#   - smooth connection for Q in [0.01, 0.03] using Eq.C12 -> RT Hermite bridge
# ============================================================

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import warnings

import numpy as np


# ------------------------------------------------------------
# 0. Configuration
# ------------------------------------------------------------
MODULE_DIR = Path(__file__).resolve().parent

# This matches the current repository layout:
#   repo_root/
#     data/
#     notebooks/
#     src/
REPO_ROOT = Path(__file__).resolve().parents[1]

# Base directory that contains ``Q_table.inp`` and/or the folders
# ``StokesI_emergent`` / ``StokesQ_emergent``.
DATA_DIR = str(REPO_ROOT / "data")
DEFAULT_I_DIRNAME = "StokesI_emergent"
DEFAULT_Q_DIRNAME = "StokesQ_emergent"

# Use Eq.(C12) Q-table in this thin regime.
TAU_EQ10_USE_MIN = 1e-4

MIN_VALID_TAU_MAX = 0.0
MAX_VALID_OMEGA = 0.9
MIN_VALID_INC_DEG = 0.0
MAX_VALID_INC_DEG = 89.0


@dataclass(frozen=True)
class StokesInterpolationContext:
    """
    Explicit data bundle for the interpolation module.

    Passing this object to the public API avoids module-level mutable state.
    """

    tables: dict
    interp_tables: dict
    q_thin_table: dict | None
    base_dir: Path | None = None
    q_table_path: Path | None = None


# ============================================================
# Helpers for validation and explicit context passing
# ============================================================

def _resolve_inc_deg(inc=None, inc_deg=None):
    """Resolve the public inclination argument."""
    if inc is None and inc_deg is None:
        raise TypeError("Pass inc or inc_deg.")

    if inc_deg is None:
        inc_deg = inc
    elif inc is not None and not np.isclose(float(inc), float(inc_deg)):
        raise ValueError("inc and inc_deg were both given but do not match.")

    return float(inc_deg)


def _validate_public_inputs(tau_max, omega, inc_deg):
    """Validate user-supplied public inputs."""
    tau_max = float(tau_max)
    omega = float(omega)
    inc_deg = float(inc_deg)

    if tau_max < MIN_VALID_TAU_MAX:
        raise ValueError(f"tau_max must satisfy tau_max >= {MIN_VALID_TAU_MAX}.")

    if not (0.0 <= omega <= MAX_VALID_OMEGA):
        raise ValueError(
            "omega is outside the validated range for the interpolation tables. "
            f"Use 0.0 <= omega <= {MAX_VALID_OMEGA}."
        )

    if not (MIN_VALID_INC_DEG <= inc_deg <= MAX_VALID_INC_DEG):
        raise ValueError(
            "inc must satisfy "
            f"{MIN_VALID_INC_DEG} <= inc <= {MAX_VALID_INC_DEG} degrees."
        )

    return tau_max, omega, inc_deg


def _require_context(context=None, data_dir=DATA_DIR, idir=None, qdir=None, q_table_path=None):
    """
    Return an explicit interpolation context.

    If ``context`` is not given, the tables are loaded on demand. Reusing a
    previously created context is recommended for repeated evaluations.
    """
    if context is not None:
        return context
    return setup_tables(data_dir=data_dir, idir=idir, qdir=qdir, q_table_path=q_table_path)


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

    - ``Q_table.inp`` + ``StokesI_emergent/`` + ``StokesQ_emergent/``
    - ``Q_table.inp`` + flat files such as ``I_tau0_1_omega0_1.inp``
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
    Find ``Q_table.inp``.
    """
    for base_dir in _candidate_base_dirs(data_dir):
        candidate = base_dir / "Q_table.inp"
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Q_table.inp not found in any candidate base directory.")


def load_emergent_file(path):
    """
    Read one new-format split file such as

    - ``I_tau0_1_omega0_1.inp``
    - ``Q_tau0_1_omega0_1.inp``

    Returns
    -------
    dict
        ``{"kind", "omega", "mu", "tau_max", "data"}``
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
    Build a dict keyed by ``(tau_max, omega)`` from split emergent files.
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


def _collect_emergent_tables_from_filelists(filelist_I, filelist_Q):
    """
    Rebuild the interpolation-ready table structure from explicit file lists.
    """
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
                raise ValueError(f"mu grid mismatch across omega files at tau={tau}")

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


def collect_emergent_tables(data_dir=None, idir=None, qdir=None):
    """
    Scan the new-format emergent files and rebuild the old interpolation-ready
    table structure.
    """
    _, filelist_I, filelist_Q = _resolve_emergent_filelists(
        data_dir=data_dir,
        idir=idir,
        qdir=qdir,
    )
    return _collect_emergent_tables_from_filelists(filelist_I, filelist_Q)


# ============================================================
# 2. Build interpolation-friendly structure
# ============================================================

def build_interp_tables(tables):
    """
    Convert collected tables into a structure for interpolation.
    """
    tau_list = sorted(tables.keys())
    if len(tau_list) < 3:
        raise ValueError(
            "At least three tau grid points are required for the cubic-Hermite "
            "tau interpolation used by this module."
        )

    tau_grid = np.array(tau_list, dtype=np.float64)

    ref = tables[tau_list[0]]["I"]
    omega_ref = np.asarray(ref["omega"], dtype=np.float64)

    per_tau = []
    for tau in tau_list:
        bucket = tables[tau]
        rec_I = bucket["I"]
        rec_Q = bucket["Q"]

        if not (
            np.allclose(rec_I["omega"], omega_ref)
            and np.allclose(rec_Q["omega"], omega_ref)
        ):
            raise ValueError(f"omega grid mismatch at tau={tau}")

        if not np.allclose(rec_I["mu"], rec_Q["mu"]):
            raise ValueError(f"mu grid mismatch between I and Q at tau={tau}")

        Idata = np.asarray(rec_I["data"], dtype=np.float32)
        Qdata = np.asarray(rec_Q["data"], dtype=np.float32)

        if Idata.shape != Qdata.shape:
            raise ValueError(
                f"data shape mismatch at tau={tau}: I{Idata.shape}, Q{Qdata.shape}"
            )

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


def load_Q_thin_table(path):
    """
    Load the small-tau Stokes-Q table computed from Eq. (C12).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Q thin table not found: {path}")

    with path.open() as f:
        _ = f.readline()  # comment line
        shape_line = f.readline().strip()
        n_mu, n_omega, n_tau = map(int, shape_line.split())

        line = f.readline().strip()
        if not line.startswith("# mu"):
            raise ValueError(f"{path}: expected '# mu' line, got {line!r}")
        mu_vals = np.fromstring(f.readline(), sep=" ")
        if mu_vals.size != n_mu:
            raise ValueError(f"{path}: mu grid size mismatch")

        line = f.readline().strip()
        if not line.startswith("# omega"):
            raise ValueError(f"{path}: expected '# omega' line, got {line!r}")
        omega_vals = np.fromstring(f.readline(), sep=" ")
        if omega_vals.size != n_omega:
            raise ValueError(f"{path}: omega grid size mismatch")

        line = f.readline().strip()
        if not line.startswith("# tau"):
            raise ValueError(f"{path}: expected '# tau' line, got {line!r}")
        tau_vals = np.fromstring(f.readline(), sep=" ")
        if tau_vals.size != n_tau:
            raise ValueError(f"{path}: tau grid size mismatch")

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


def setup_tables(data_dir=DATA_DIR, idir=None, qdir=None, q_table_path=None):
    """
    Read new-format Stokes-I/Q files and build an explicit interpolation
    context.

    Parameters
    ----------
    data_dir : str or Path, optional
        Base directory that contains ``Q_table.inp`` and either
        ``StokesI_emergent`` / ``StokesQ_emergent`` or flat ``I_tau...`` /
        ``Q_tau...`` files.
    idir, qdir : str or Path, optional
        Explicit I/Q directories. When given, these override automatic path
        discovery.
    q_table_path : str or Path or None, optional
        Explicit path to ``Q_table.inp``. When omitted, the path is resolved
        automatically.
    """
    base_dir, filelist_I, filelist_Q = _resolve_emergent_filelists(
        data_dir=data_dir,
        idir=idir,
        qdir=qdir,
    )
    tables = _collect_emergent_tables_from_filelists(filelist_I, filelist_Q)
    interp_tables = build_interp_tables(tables)

    if q_table_path is not None:
        q_table_path = Path(q_table_path)
    else:
        search_root = data_dir if data_dir is not None else base_dir
        try:
            q_table_path = _resolve_q_table_path(search_root)
        except FileNotFoundError:
            q_table_path = None

    q_thin_table = None
    if q_table_path is not None:
        q_thin_table = load_Q_thin_table(q_table_path)

    return StokesInterpolationContext(
        tables=tables,
        interp_tables=interp_tables,
        q_thin_table=q_thin_table,
        base_dir=Path(base_dir) if base_dir is not None else None,
        q_table_path=Path(q_table_path) if q_table_path is not None else None,
    )


# ============================================================
# 3. Low-level interpolation in omega, mu
# ============================================================

def _interp_along_mu(mu, mu_grid, data_omega_mu):
    """
    Linear interpolation along mu for each omega.
    Returns shape ``(N_omega,)``.
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
    2D interpolation in ``(omega, mu)`` using linear interpolation in both
    directions.
    """
    vals_vs_omega = _interp_along_mu(mu, mu_grid, data_omega_mu)
    return _interp1d_omega(omega, omega_grid, vals_vs_omega)


# ============================================================
# 4. Smooth interpolation along tau (cubic Hermite)
# ============================================================

def interp_stokes_scalar_smooth(tau_max, omega, mu, context):
    """
    Low-level interpolator:
      ``(tau_max, omega, mu) -> kind ("I" or "Q")``
    """
    tau_max = float(tau_max)
    omega = float(omega)
    mu = float(mu)

    # almost pure absorption
    if abs(omega) < 1e-6:
        mu_eff = max(mu, 1e-3)
        return {
            "I": float(1.0 - np.exp(-tau_max / mu_eff)),
            "Q": 0.0,
        }

    tau_grid = context.interp_tables["tau_grid"]
    omega_grid = context.interp_tables["omega_grid"]
    per_tau = context.interp_tables["per_tau"]

    tau_min = float(tau_grid[0])
    tau_max_grid = float(tau_grid[-1])
    tau_c = float(np.clip(tau_max, tau_min, tau_max_grid))

    N = len(tau_grid)
    idx_hi = int(np.searchsorted(tau_grid, tau_c, side="right"))
    idx_hi = max(1, min(idx_hi, N - 1))
    idx_lo = idx_hi - 1

    def _value_for_kind(bucket, kind):
        return _interp2d(omega, mu, omega_grid, bucket["mu_grid"], bucket[kind])

    # Left edge interval
    if idx_lo == 0:
        k0, k1, k2 = 0, 1, 2
        tau0 = float(tau_grid[k0])
        tau1 = float(tau_grid[k1])
        tau2 = float(tau_grid[k2])

        rec0 = per_tau[k0]
        rec1 = per_tau[k1]
        rec2 = per_tau[k2]

        out = {}
        for kind in ("I", "Q"):
            y0 = _value_for_kind(rec0, kind)
            y1 = _value_for_kind(rec1, kind)
            y2 = _value_for_kind(rec2, kind)

            d01 = (y1 - y0) / (tau1 - tau0)
            d12 = (y2 - y1) / (tau2 - tau1)

            m0 = d01
            m1 = 0.5 * (d01 + d12)

            s = (tau_c - tau0) / (tau1 - tau0)
            h00 = 2.0 * s**3 - 3.0 * s**2 + 1.0
            h10 = s**3 - 2.0 * s**2 + s
            h01 = -2.0 * s**3 + 3.0 * s**2
            h11 = s**3 - s**2
            dt = tau1 - tau0

            out[kind] = float(h00 * y0 + h10 * dt * m0 + h01 * y1 + h11 * dt * m1)

        return out

    # Right edge interval
    if idx_hi == N - 1:
        k0, k1, k2 = N - 3, N - 2, N - 1
        tau0 = float(tau_grid[k0])
        tau1 = float(tau_grid[k1])
        tau2 = float(tau_grid[k2])

        rec0 = per_tau[k0]
        rec1 = per_tau[k1]
        rec2 = per_tau[k2]

        out = {}
        for kind in ("I", "Q"):
            y0 = _value_for_kind(rec0, kind)
            y1 = _value_for_kind(rec1, kind)
            y2 = _value_for_kind(rec2, kind)

            d01 = (y1 - y0) / (tau1 - tau0)
            d12 = (y2 - y1) / (tau2 - tau1)

            m1 = 0.5 * (d01 + d12)
            m2 = d12

            s = (tau_c - tau1) / (tau2 - tau1)
            h00 = 2.0 * s**3 - 3.0 * s**2 + 1.0
            h10 = s**3 - 2.0 * s**2 + s
            h01 = -2.0 * s**3 + 3.0 * s**2
            h11 = s**3 - s**2
            dt = tau2 - tau1

            out[kind] = float(h00 * y1 + h10 * dt * m1 + h01 * y2 + h11 * dt * m2)

        return out

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

    out = {}
    for kind in ("I", "Q"):
        y0 = _value_for_kind(rec0, kind)
        y1 = _value_for_kind(rec1, kind)
        y2 = _value_for_kind(rec2, kind)
        y3 = _value_for_kind(rec3, kind)

        d10 = (y1 - y0) / (tau1 - tau0)
        d21 = (y2 - y1) / (tau2 - tau1)
        d32 = (y3 - y2) / (tau3 - tau2)

        m1 = 0.5 * (d21 + d10)
        m2 = 0.5 * (d32 + d21)

        s = (tau_c - tau1) / (tau2 - tau1)
        h00 = 2.0 * s**3 - 3.0 * s**2 + 1.0
        h10 = s**3 - 2.0 * s**2 + s
        h01 = -2.0 * s**3 + 3.0 * s**2
        h11 = s**3 - s**2
        dt = tau2 - tau1

        out[kind] = float(h00 * y1 + h10 * dt * m1 + h01 * y2 + h11 * dt * m2)

    return out


# ============================================================
# 5. Public pure-interpolation API (no thin/thick patches)
# ============================================================

def interpolate_stokes(
    tau_max,
    omega,
    inc=None,
    *,
    inc_deg=None,
    context=None,
    data_dir=DATA_DIR,
    idir=None,
    qdir=None,
    q_table_path=None,
):
    """
    Pure interpolation from RT tables:
        ``(tau_max, omega, inc [deg]) -> (I, Q)``

    No optically thin / thick patches are applied.
    """
    inc_deg = _resolve_inc_deg(inc=inc, inc_deg=inc_deg)
    tau_max, omega, inc_deg = _validate_public_inputs(tau_max=tau_max, omega=omega, inc_deg=inc_deg)
    context = _require_context(
        context=context,
        data_dir=data_dir,
        idir=idir,
        qdir=qdir,
        q_table_path=q_table_path,
    )

    mu = float(np.cos(np.deg2rad(inc_deg)))
    values = interp_stokes_scalar_smooth(tau_max, omega, mu, context=context)
    return float(values["I"]), float(values["Q"])


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


def Q_thin_scalar(tau_max, omega, inc_deg, context):
    """
    Optically thin fallback approximation for Stokes Q.
    """
    tau_max = float(tau_max)
    omega = float(omega)
    inc_deg = float(inc_deg)

    if tau_max <= 0.0 or omega <= 0.0 or omega >= 1.0:
        return 0.0

    tau_grid = context.interp_tables["tau_grid"]
    tau_ref = float(tau_grid[0])

    mu = float(np.cos(np.deg2rad(inc_deg)))
    mu = float(np.clip(mu, 1e-10, 1.0))

    _, Q_ref = interpolate_stokes(tau_ref, omega, inc=inc_deg, context=context)

    base_ref = ((1.0 - mu**2) / mu) * omega * (1.0 - omega) * (tau_ref**2)
    eps = 1e-30
    if abs(base_ref) < eps:
        return float(Q_ref * (tau_max / tau_ref) ** 2)

    c_Q = Q_ref / base_ref
    base_t = ((1.0 - mu**2) / mu) * omega * (1.0 - omega) * (tau_max**2)
    return float(c_Q * base_t)


def Q_eqC12_table_scalar(tau_max, omega, inc_deg, context):
    """
    ``Q/B_nu`` from the small-tau table computed using Eq. (C12).
    Trilinear interpolation in ``(mu, omega, tau)``.
    """
    if context.q_thin_table is None:
        raise RuntimeError("Q thin table is unavailable in the supplied context.")

    tau_max = float(tau_max)
    omega = float(omega)
    inc_deg = float(inc_deg)

    mu = float(np.cos(np.deg2rad(inc_deg)))
    mu = float(np.clip(mu, 0.0, 1.0))

    mu_grid = context.q_thin_table["mu_grid"]
    omega_grid = context.q_thin_table["omega_grid"]
    tau_grid = context.q_thin_table["tau_grid"]
    data = context.q_thin_table["data"]  # (n_mu, n_omega, n_tau)

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


def Q_C8_scalar(tau_max, omega, inc_deg, context):
    """
    Extremely thin limit for Stokes Q using an Eq. (C8)-like form.
    """
    if context.q_thin_table is None:
        return Q_thin_scalar(tau_max, omega, inc_deg, context=context)

    tau_max = float(tau_max)
    omega = float(omega)
    inc_deg = float(inc_deg)

    if tau_max <= 0.0 or omega <= 0.0 or omega >= 1.0:
        return 0.0

    tau_grid_thin = context.q_thin_table["tau_grid"]
    tau_ref = float(tau_grid_thin[0])

    mu = float(np.cos(np.deg2rad(inc_deg)))
    mu = float(np.clip(mu, 1e-10, 1.0))

    Q_ref = Q_eqC12_table_scalar(tau_ref, omega, inc_deg, context=context)

    base_ref = ((1.0 - mu**2) / mu) * omega * (1.0 - omega) * (tau_ref**2)
    eps = 1e-30
    if abs(base_ref) < eps:
        return float(Q_ref * (tau_max / tau_ref) ** 2)

    c_q = Q_ref / base_ref
    base_t = ((1.0 - mu**2) / mu) * omega * (1.0 - omega) * (tau_max**2)
    return float(c_q * base_t)


def _deriv_at_x1_quadratic(x0, x1, x2, y0, y1, y2):
    """
    Derivative at x1 of the quadratic through (x0,y0),(x1,y1),(x2,y2).
    """
    return (
        y0 * (x1 - x2) / ((x0 - x1) * (x0 - x2))
        + y1 * (2 * x1 - x0 - x2) / ((x1 - x0) * (x1 - x2))
        + y2 * (x1 - x0) / ((x2 - x0) * (x2 - x1))
    )


def Q_bridge_small_tau(tau_max, omega, inc_deg, context):
    """
    Smoothly connect Q from the Eq. (C12) thin table to RT tables
    in the first RT interval [tau0, tau1] using cubic Hermite.
    """
    if context.q_thin_table is None:
        raise RuntimeError("Q thin table is unavailable in the supplied context.")

    tau_max = float(tau_max)
    omega = float(omega)
    inc_deg = float(inc_deg)

    tau_grid = context.interp_tables["tau_grid"]
    tau0 = float(tau_grid[0])
    tau1 = float(tau_grid[1])
    tau2 = float(tau_grid[2])

    if not (tau0 <= tau_max <= tau1):
        raise ValueError("Q_bridge_small_tau is only valid for tau in [tau0, tau1].")

    # Left endpoint from Eq.(C12) side: value at tau0
    Q0 = float(Q_eqC12_table_scalar(tau0, omega, inc_deg, context=context))

    # Left slope m0 from Eq.(C12): use tau just below tau0 (backward diff)
    tg = np.asarray(context.q_thin_table["tau_grid"], dtype=float)
    k = int(np.searchsorted(tg, tau0, side="right") - 1)

    if k <= 0:
        tA = float(tg[0])
        tB = float(tg[1])
        QA = float(Q_eqC12_table_scalar(tA, omega, inc_deg, context=context))
        QB = float(Q_eqC12_table_scalar(tB, omega, inc_deg, context=context))
        m0 = (QB - QA) / (tB - tA)
    else:
        t_prev = float(tg[k - 1])
        Q_prev = float(Q_eqC12_table_scalar(t_prev, omega, inc_deg, context=context))
        m0 = (Q0 - Q_prev) / (tau0 - t_prev)

    # Right endpoint from RT side
    _, Q1 = interpolate_stokes(tau1, omega, inc=inc_deg, context=context)

    # RT-side derivative at tau1 (quadratic fit using tau0,tau1,tau2)
    _, Q0_rt = interpolate_stokes(tau0, omega, inc=inc_deg, context=context)
    _, Q2_rt = interpolate_stokes(tau2, omega, inc=inc_deg, context=context)
    m1 = _deriv_at_x1_quadratic(tau0, tau1, tau2, Q0_rt, Q1, Q2_rt)

    # Hermite in [tau0, tau1]
    s = (tau_max - tau0) / (tau1 - tau0)
    h00 = 2.0 * s**3 - 3.0 * s**2 + 1.0
    h10 = s**3 - 2.0 * s**2 + s
    h01 = -2.0 * s**3 + 3.0 * s**2
    h11 = s**3 - s**2
    dt = tau1 - tau0

    Q_bridge = h00 * Q0 + h10 * dt * m0 + h01 * Q1 + h11 * dt * m1
    return float(Q_bridge)


def Q_bridge_tau0_tau1_fallback(tau_max, omega, inc_deg, context):
    """
    Fallback bridge when the thin Q table is unavailable.
    """
    tau_max = float(tau_max)
    omega = float(omega)
    inc_deg = float(inc_deg)

    tau_grid = context.interp_tables["tau_grid"]
    tau0 = float(tau_grid[0])
    tau1 = float(tau_grid[1])
    tau2 = float(tau_grid[2])

    if tau_max <= tau0:
        return float(Q_thin_scalar(tau_max, omega, inc_deg, context=context))
    if tau_max >= tau1:
        _, Q = interpolate_stokes(tau_max, omega, inc=inc_deg, context=context)
        return float(Q)

    # left: thin value and slope (assume ~tau^2)
    Q0 = float(Q_thin_scalar(tau0, omega, inc_deg, context=context))
    m0 = 0.0 if tau0 == 0.0 else (2.0 * Q0 / tau0)

    # right: RT value and slope
    _, Q1 = interpolate_stokes(tau1, omega, inc=inc_deg, context=context)
    _, Q0_rt = interpolate_stokes(tau0, omega, inc=inc_deg, context=context)
    _, Q2_rt = interpolate_stokes(tau2, omega, inc=inc_deg, context=context)
    m1 = _deriv_at_x1_quadratic(tau0, tau1, tau2, Q0_rt, Q1, Q2_rt)

    # Hermite
    s = (tau_max - tau0) / (tau1 - tau0)
    h00 = 2.0 * s**3 - 3.0 * s**2 + 1.0
    h10 = s**3 - 2.0 * s**2 + s
    h01 = -2.0 * s**3 + 3.0 * s**2
    h11 = s**3 - s**2
    dt = tau1 - tau0

    return float(h00 * Q0 + h10 * dt * m0 + h01 * Q1 + h11 * dt * m1)


# ============================================================
# 7. High-level emergent API with thin/thick patches
# ============================================================

def emergent_stokes(
    tau_max,
    omega,
    inc=None,
    *,
    inc_deg=None,
    context=None,
    data_dir=DATA_DIR,
    idir=None,
    qdir=None,
    q_table_path=None,
):
    """
    High-level API:
        ``(tau_max, omega, inc [deg]) -> (I, Q)``

    This function applies:

    - ``omega = 0``: pure absorption ``I``, ``Q = 0``
    - tau below RT min: thin patches
    - first RT interval ``[tau0, tau1]``: Q bridged smoothly
    - tau above RT max: saturation at ``tau_max_table`` (RT)
    - otherwise: pure RT interpolation
    """
    inc_deg = _resolve_inc_deg(inc=inc, inc_deg=inc_deg)
    tau_max, omega, inc_deg = _validate_public_inputs(tau_max=tau_max, omega=omega, inc_deg=inc_deg)
    context = _require_context(
        context=context,
        data_dir=data_dir,
        idir=idir,
        qdir=qdir,
        q_table_path=q_table_path,
    )

    if tau_max == 0.0:
        return 0.0, 0.0

    tau_grid = context.interp_tables["tau_grid"]
    tau_min_table = float(tau_grid[0])
    tau_max_table = float(tau_grid[-1])

    # first RT interval [tau0, tau1]
    tau0 = float(tau_grid[0])
    tau1 = float(tau_grid[1])

    # omega = 0: pure absorption solution
    if np.isclose(omega, 0.0):
        I = analytic_thin_I_scalar(tau_max, omega, inc_deg)
        Q = 0.0
        return float(I), float(Q)

    # 1) optically thin region below RT tables
    if tau_max < tau_min_table:
        I = analytic_thin_I_scalar(tau_max, omega, inc_deg)

        if context.q_thin_table is not None:
            tau_grid_thin = np.asarray(context.q_thin_table["tau_grid"], dtype=float)
            tau_min_thin = float(tau_grid_thin[0])
            tau_max_thin = float(tau_grid_thin[-1])

            if tau_max < TAU_EQ10_USE_MIN:
                Q = Q_C8_scalar(tau_max, omega, inc_deg, context=context)
            else:
                tau_for_eq10 = float(np.clip(tau_max, tau_min_thin, tau_max_thin))
                Q = Q_eqC12_table_scalar(tau_for_eq10, omega, inc_deg, context=context)
        else:
            Q = Q_thin_scalar(tau_max, omega, inc_deg, context=context)

        return float(I), float(Q)

    # 2) first RT interval: bridge Q smoothly
    if tau0 <= tau_max <= tau1:
        I, _ = interpolate_stokes(tau_max, omega, inc=inc_deg, context=context)

        if context.q_thin_table is not None:
            Q = Q_bridge_small_tau(tau_max, omega, inc_deg, context=context)
        else:
            Q = Q_bridge_tau0_tau1_fallback(tau_max, omega, inc_deg, context=context)

        return float(I), float(Q)

    # 3) very thick: saturate at tau_max_table
    if tau_max > tau_max_table:
        warnings.warn(
            "tau_max exceeds the RT-table range; returning the saturated value "
            f"at tau_max = {tau_max_table}.",
            RuntimeWarning,
            stacklevel=2,
        )
        I_sat, Q_sat = interpolate_stokes(tau_max_table, omega, inc=inc_deg, context=context)
        return float(I_sat), float(Q_sat)

    # 4) intermediate: pure RT interpolation
    I, Q = interpolate_stokes(tau_max, omega, inc=inc_deg, context=context)
    return float(I), float(Q)


def emergent_polarization(
    tau_max,
    omega,
    inc=None,
    *,
    inc_deg=None,
    context=None,
    data_dir=DATA_DIR,
    idir=None,
    qdir=None,
    q_table_path=None,
    mode="Q_over_I",
):
    """
    Convenience wrapper computed from ``(I, Q)`` without any PF table.

    mode
    ----
    - ``"Q_over_I"``: return ``Q/I``
    - ``"abs_Q_over_I"``: return ``|Q|/I``
    - ``"signed_abs"``: return ``sign(Q) * |Q|/I``
    """
    I, Q = emergent_stokes(
        tau_max=tau_max,
        omega=omega,
        inc=inc,
        inc_deg=inc_deg,
        context=context,
        data_dir=data_dir,
        idir=idir,
        qdir=qdir,
        q_table_path=q_table_path,
    )
    I = float(I)
    Q = float(Q)
    if I == 0.0:
        return 0.0

    if mode == "Q_over_I":
        return float(Q / I)
    if mode == "abs_Q_over_I":
        return float(abs(Q) / I)
    if mode == "signed_abs":
        return float(np.sign(Q) * (abs(Q) / I))
    raise ValueError(f"unknown mode: {mode}")


# Backward-compatible alias with a clearer name
load_stokes_context = setup_tables
