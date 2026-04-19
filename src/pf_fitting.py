"""Fitting utilities for polarization fraction (PF)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import lmfit
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import numpy as np
from scipy.interpolate import interp1d

# =========================
# Constants
# =========================
EPS = 1e-12
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_INDIR_I = PROJECT_ROOT / "data" / "StokesI_emergent"
DEFAULT_INDIR_Q = PROJECT_ROOT / "data" / "StokesQ_emergent"

MIN_VALID_INC_DEG = 0.0
MAX_VALID_INC_DEG = 80.0
MAX_VALID_OMEGA = 0.9


@dataclass(frozen=True)
class PFData:
    """Container for the Stokes-I/Q tables used in PF fitting."""

    tau_max_values: np.ndarray
    omega_values: np.ndarray
    mu_pos_grid: np.ndarray
    I_surface: np.ndarray
    Q_surface: np.ndarray
    indir_I: Path
    indir_Q: Path


def _resolve_angle_deg(inc=None, angle_deg=None, *, default=45.0):
    """Resolve the public inclination argument."""
    if inc is None and angle_deg is None:
        return float(default)

    if angle_deg is None:
        angle_deg = inc
    elif inc is not None and not np.isclose(float(inc), float(angle_deg)):
        raise ValueError("inc and angle_deg were both given but do not match.")

    return float(angle_deg)


def _validate_fit_inputs(angle_deg, degree):
    """Validate public inputs for the fitting workflow."""
    if not (MIN_VALID_INC_DEG <= float(angle_deg) <= MAX_VALID_INC_DEG):
        raise ValueError(
            "inc must satisfy "
            f"{MIN_VALID_INC_DEG} <= inc <= {MAX_VALID_INC_DEG} degrees."
        )

    if int(degree) != degree or degree < 0:
        raise ValueError("degree must be a non-negative integer.")


def _validate_evaluation_inputs(omega, tau_max):
    """Validate user-supplied evaluation inputs."""
    omega = np.asarray(omega, dtype=float)
    tau_max = np.asarray(tau_max, dtype=float)

    if np.any((omega <= 0.0) | (omega > MAX_VALID_OMEGA)):
        raise ValueError(
            "omega is outside the validated range for this PF fitting formula. "
            f"Use 0.0 < omega <= {MAX_VALID_OMEGA}."
        )

    if np.any(tau_max < 0.0):
        raise ValueError("tau_max must satisfy tau_max >= 0.0.")

    return omega, tau_max


# =========================
# Read .inp files
# =========================
def read_single_inp(filepath):
    """
    Read a single .inp file and return
    omega, tau_max, mu_pos, and the Stokes value.
    """
    filepath = Path(filepath)
    omega = None
    tau_max = None
    n_mu = None

    with filepath.open("r", encoding="utf-8") as f:
        header_lines = []
        data_lines = []

        for line in f:
            if line.startswith("#"):
                header_lines.append(line.strip())
            elif line.strip():
                data_lines.append(line)

    for line in header_lines:
        if line.startswith("# omega ="):
            omega = float(line.split("=", 1)[1].strip())
        elif line.startswith("# tau_max ="):
            tau_max = float(line.split("=", 1)[1].strip())
        elif line.startswith("# n_mu ="):
            n_mu = int(line.split("=", 1)[1].strip())

    if omega is None:
        raise ValueError(f"{filepath}: omega not found in header")
    if tau_max is None:
        raise ValueError(f"{filepath}: tau_max not found in header")
    if n_mu is None:
        raise ValueError(f"{filepath}: n_mu not found in header")
    if not data_lines:
        raise ValueError(f"{filepath}: no data rows found")

    arr = np.loadtxt(data_lines)
    if arr.ndim == 1:
        arr = arr.reshape(1, 2)

    mu_pos = np.asarray(arr[:, 0], dtype=float)
    stokes_val = np.asarray(arr[:, 1], dtype=float)

    if len(mu_pos) != n_mu:
        raise ValueError(f"{filepath}: header n_mu and actual row count do not match")

    return float(omega), float(tau_max), mu_pos, stokes_val


# =========================
# Helper function to build a record dictionary from .inp files
# =========================
def _build_record_map(filelist, label):
    """Build a dict keyed by (omega, tau_max)."""
    records = {}

    for filepath in filelist:
        omega, tau_max, mu_pos, stokes_val = read_single_inp(filepath)
        key = (float(omega), float(tau_max))

        if key in records:
            raise ValueError(
                f"Duplicate {label} file found for omega={omega}, tau_max={tau_max}: {filepath}"
            )

        records[key] = {
            "filepath": Path(filepath),
            "omega": float(omega),
            "tau_max": float(tau_max),
            "mu_pos": mu_pos,
            "stokes_val": stokes_val,
        }

    return records


# =========================
# Load all I/Q .inp files into an explicit data bundle
# =========================
def load_IQ_inp(indir_I=DEFAULT_INDIR_I, indir_Q=DEFAULT_INDIR_Q):
    """
    Read Stokes-I and Stokes-Q .inp files and reconstruct the fitting arrays.

    Notes
    -----
    ``omega = 0`` is excluded from the PF fitting because PF is always zero
    there and ``StokesQ_emergent`` does not contain ``omega = 0`` files.
    """
    indir_I = Path(indir_I)
    indir_Q = Path(indir_Q)

    filelist_I = sorted(indir_I.glob("*.inp"))
    filelist_Q = sorted(indir_Q.glob("*.inp"))

    if not filelist_I:
        raise FileNotFoundError(f"no inp files found in {indir_I}")
    if not filelist_Q:
        raise FileNotFoundError(f"no inp files found in {indir_Q}")

    records_I_all = _build_record_map(filelist_I, "Stokes I")
    records_Q = _build_record_map(filelist_Q, "Stokes Q")

    keys_Q = {key for key in records_Q if not np.isclose(key[0], 0.0)}
    keys_I = {key for key in records_I_all if key in keys_Q and not np.isclose(key[0], 0.0)}

    missing_in_Q = sorted(keys_I - keys_Q)
    missing_in_I = sorted(keys_Q - keys_I)

    if missing_in_Q or missing_in_I:
        raise ValueError(
            "Stokes I and Stokes Q grids do not match for nonzero omega.\n"
            f"Missing in {indir_Q}: {missing_in_Q}\n"
            f"Missing in {indir_I}: {missing_in_I}"
        )

    tau_max_values = np.array(sorted({key[1] for key in keys_I}), dtype=float)
    omega_values = np.array(sorted({key[0] for key in keys_I}), dtype=float)

    n_tau = len(tau_max_values)
    n_omega = len(omega_values)

    tau_to_idx = {v: i for i, v in enumerate(tau_max_values)}
    omega_to_idx = {v: i for i, v in enumerate(omega_values)}

    mu_pos_grid = np.full((n_tau,), None, dtype=object)
    I_surface = np.empty((n_omega, n_tau), dtype=object)
    Q_surface = np.empty((n_omega, n_tau), dtype=object)
    filled = np.zeros((n_omega, n_tau), dtype=bool)

    for key in sorted(keys_I):
        rec_I = records_I_all[key]
        rec_Q = records_Q[key]

        i_tau = tau_to_idx[rec_I["tau_max"]]
        i_omega = omega_to_idx[rec_I["omega"]]

        if not np.array_equal(rec_I["mu_pos"], rec_Q["mu_pos"]):
            same_len = len(rec_I["mu_pos"]) == len(rec_Q["mu_pos"])
            same_grid = np.allclose(rec_I["mu_pos"], rec_Q["mu_pos"])
            if not (same_len and same_grid):
                raise ValueError(
                    f"Inconsistent mu grid between I and Q for omega={rec_I['omega']}, "
                    f"tau_max={rec_I['tau_max']}"
                )

        if mu_pos_grid[i_tau] is None:
            mu_pos_grid[i_tau] = rec_I["mu_pos"]
        else:
            same_len = len(mu_pos_grid[i_tau]) == len(rec_I["mu_pos"])
            same_grid = np.allclose(mu_pos_grid[i_tau], rec_I["mu_pos"])
            if not (same_len and same_grid):
                raise ValueError(
                    f"Inconsistent mu grid for tau_max={rec_I['tau_max']} "
                    f"between files. File: {rec_I['filepath']}"
                )

        I_surface[i_omega, i_tau] = rec_I["stokes_val"]
        Q_surface[i_omega, i_tau] = rec_Q["stokes_val"]
        filled[i_omega, i_tau] = True

    if not np.all(filled):
        missing = np.argwhere(~filled)
        raise ValueError(f"Some (omega, tau_max) pairs are missing: {missing}")

    return PFData(
        tau_max_values=tau_max_values,
        omega_values=omega_values,
        mu_pos_grid=mu_pos_grid,
        I_surface=I_surface,
        Q_surface=Q_surface,
        indir_I=indir_I,
        indir_Q=indir_Q,
    )


def _coerce_pf_surface_inputs(data=None, mu_pos_grid=None, I_surface=None, Q_surface=None):
    """
    Accept either a ``PFData`` bundle or the raw arrays used in the
    original public API.
    """
    if data is not None:
        return data.mu_pos_grid, data.I_surface, data.Q_surface

    if mu_pos_grid is None or I_surface is None or Q_surface is None:
        raise TypeError(
            "Pass either data=<PFData> or mu_pos_grid, I_surface, and Q_surface."
        )

    return mu_pos_grid, I_surface, Q_surface


# =========================
# Interpolate PF to a requested angle
# =========================
def get_PF_at_angle_fast(
    inc=45.0,
    mu_pos_grid=None,
    I_surface=None,
    Q_surface=None,
    data=None,
    angle_deg=None,
):
    """
    Interpolate Stokes I/Q onto the requested viewing angle and return PF=Q/I.
    """
    angle_deg = _resolve_angle_deg(inc=inc, angle_deg=angle_deg)
    _validate_fit_inputs(angle_deg=angle_deg, degree=0)
    mu_pos_grid, I_surface, Q_surface = _coerce_pf_surface_inputs(
        data=data,
        mu_pos_grid=mu_pos_grid,
        I_surface=I_surface,
        Q_surface=Q_surface,
    )

    mu = np.cos(np.deg2rad(angle_deg))

    n_omega = I_surface.shape[0]
    n_tau = I_surface.shape[1]
    pf_angle = np.zeros((n_omega, n_tau))

    for omega_idx in range(n_omega):
        for tau_idx in range(n_tau):
            mu_grid = mu_pos_grid[tau_idx]
            I_grid = I_surface[omega_idx, tau_idx]
            Q_grid = Q_surface[omega_idx, tau_idx]

            I_interp = interp1d(mu_grid, I_grid, kind="cubic")
            Q_interp = interp1d(mu_grid, Q_grid, kind="cubic")

            I_val = float(I_interp(mu))
            Q_val = float(Q_interp(mu))
            pf_angle[omega_idx, tau_idx] = Q_val / I_val

    return pf_angle


# =========================
# PF model
# =========================
def model_func_PF(tau_max, A_PF, B_PF, PF_conv, tau_PF):
    """PF fitting formula."""
    tau_max = np.asarray(tau_max)
    return (
        A_PF * tau_max**B_PF * np.exp(-((tau_max / tau_PF) ** 0.8))
        + PF_conv * (1.0 - np.exp(-((tau_max / tau_PF) ** 0.8)))
    )


# =========================
# Convert a polynomial to a LaTeX string
# =========================
def poly_to_latex(poly, var=r"\omega", precision=3):
    """Convert ``np.poly1d`` to a compact LaTeX-like string."""
    coeffs = np.array(poly.c, dtype=float)
    deg = len(coeffs) - 1
    terms = []

    for i, c in enumerate(coeffs):
        power = deg - i
        if np.isclose(c, 0.0):
            continue

        c_round = float(f"{c:.{precision}g}")

        if power == 0:
            term = f"{abs(c_round):.{precision}g}"
        elif power == 1:
            if np.isclose(abs(c_round), 1.0):
                term = f"{var}"
            else:
                term = f"{abs(c_round):.{precision}g}{var}"
        else:
            if np.isclose(abs(c_round), 1.0):
                term = f"{var}^{power}"
            else:
                term = f"{abs(c_round):.{precision}g}{var}^{power}"

        if not terms:
            terms.append(("-" if c_round < 0 else "") + term)
        else:
            terms.append((" - " if c_round < 0 else " + ") + term)

    return "".join(terms) if terms else "0"


# =========================
# Helper function that performs the fit and generates the plot
# =========================
def fit_and_plot_PF(
    inc=45.0,
    angle_deg=None,
    degree=5,
    indir_I=DEFAULT_INDIR_I,
    indir_Q=DEFAULT_INDIR_Q,
    savepath=None,
    show=True,
    data=None,
):
    """
    Perform the PF fitting and make the plot.

    Parameters
    ----------
    inc : float, optional
        Inclination angle in degrees. Use keyword arguments in user-facing
        examples.
    angle_deg : float or None, optional
        Backward-compatible alias for ``inc``. If given, it overrides ``inc``.
    degree : int, optional
        Polynomial degree for fitting the omega dependence.
    indir_I, indir_Q : str or Path, optional
        Directories containing Stokes-I and Stokes-Q .inp files. Ignored when
        ``data`` is given.
    savepath : str or Path or None, optional
        Output figure path.
    show : bool, optional
        If True, show the figure.
    data : PFData or None, optional
        Explicit data bundle returned by :func:`load_IQ_inp`. Passing this
        avoids module-level global state and repeated reloading.
    """
    angle_deg = _resolve_angle_deg(inc=inc, angle_deg=angle_deg)
    _validate_fit_inputs(angle_deg=angle_deg, degree=degree)

    if data is None:
        data = load_IQ_inp(indir_I=indir_I, indir_Q=indir_Q)

    pf_angle = get_PF_at_angle_fast(inc=angle_deg, data=data)

    my_model = lmfit.Model(model_func_PF)

    A_PF = np.zeros(len(data.omega_values))
    B_PF = np.zeros(len(data.omega_values))
    PF_conv = np.zeros(len(data.omega_values))
    tau_PF = np.zeros(len(data.omega_values))

    for omega_idx in range(len(data.omega_values)):
        ydata = pf_angle[omega_idx, :]
        weights = 1.0 / np.maximum(np.abs(ydata), EPS)

        params = my_model.make_params(A_PF=1.0, B_PF=0.5, PF_conv=1.0, tau_PF=1.0)

        results = my_model.fit(
            ydata,
            tau_max=data.tau_max_values,
            params=params,
            weights=weights,
            nan_policy="omit",
            method="least_squares",
            fit_kws={"loss": "linear"},
        )

        A_PF[omega_idx] = results.best_values["A_PF"]
        B_PF[omega_idx] = results.best_values["B_PF"]
        PF_conv[omega_idx] = results.best_values["PF_conv"]
        tau_PF[omega_idx] = results.best_values["tau_PF"]

    poly_A = np.poly1d(np.polyfit(data.omega_values, A_PF, degree))
    poly_B = np.poly1d(np.polyfit(data.omega_values, B_PF, degree))
    poly_PFconv = np.poly1d(np.polyfit(data.omega_values, PF_conv, degree))
    poly_tauPF = np.poly1d(np.polyfit(data.omega_values, tau_PF, degree))

    A_fit = poly_A(data.omega_values)
    B_fit = poly_B(data.omega_values)
    PF_conv_fit = poly_PFconv(data.omega_values)
    tau_PF_fit = poly_tauPF(data.omega_values)

    latex_A = poly_to_latex(poly_A)
    latex_B = poly_to_latex(poly_B)
    latex_PFconv = poly_to_latex(poly_PFconv)
    latex_tauPF = poly_to_latex(poly_tauPF)

    plt.rcParams.update(
        {
            "font.size": 30,
            "font.family": "serif",
            "font.serif": ["Times", "Times New Roman"] + plt.rcParams["font.serif"],
            "mathtext.fontset": "stix",
            "mathtext.rm": "Times New Roman",
            "mathtext.it": "Times New Roman:italic",
            "mathtext.bf": "Times New Roman:bold",
        }
    )
    fig, ax = plt.subplots(2, 1, figsize=(12, 24))
    fig.subplots_adjust(top=0.68, left=0.12, right=0.95, bottom=0.08, hspace=0.30)

    fig.text(0.05, 0.975, r"$\bf{Fitting\ formula\ for\ } PF$", ha="left", va="top", fontsize=20)
    fig.text(
        0.05,
        0.940,
        r"$PF = A_{\mathrm{PF}}(\omega,\mu)\tau_{\max}^{B_{\mathrm{PF}}(\omega,\mu)}"
        r"\exp\left[-\left(\frac{\tau_{\max}}{\tau_{\mathrm{PF}}(\omega,\mu)}\right)^{0.8}\right]"
        r" + PF_{\mathrm{conv}}(\omega,\mu)\left[1-\exp\left(-\left(\frac{\tau_{\max}}{\tau_{\mathrm{PF}}(\omega,\mu)}\right)^{0.8}\right)\right]$",
        ha="left",
        va="top",
        fontsize=18,
    )
    fig.text(0.05, 0.890, rf"$\bf{{Polynomial\ fitting\ formulas\ at\ }} i={angle_deg:.1f}^\circ$", ha="left", va="top", fontsize=20)
    fig.text(0.05, 0.855, rf"$A_{{\mathrm{{PF}}}}(\omega) = {latex_A}$", ha="left", va="top", fontsize=18)
    fig.text(0.05, 0.823, rf"$B_{{\mathrm{{PF}}}}(\omega) = {latex_B}$", ha="left", va="top", fontsize=18)
    fig.text(0.05, 0.791, rf"$PF_{{\mathrm{{conv}}}}(\omega) = {latex_PFconv}$", ha="left", va="top", fontsize=18)
    fig.text(0.05, 0.759, rf"$\tau_{{\mathrm{{PF}}}}(\omega) = {latex_tauPF}$", ha="left", va="top", fontsize=18)

    tau_plot = np.arange(0.0, 15.0, 0.01)

    for omega_idx in range(len(data.omega_values)):
        current_omega = data.omega_values[omega_idx]

        ax[0].scatter(
            data.tau_max_values,
            100.0 * pf_angle[omega_idx, :],
            label=fr"$\omega={current_omega}$",
            s=40,
        )
        ax[0].plot(
            tau_plot,
            100.0 * model_func_PF(
                tau_plot,
                A_fit[omega_idx],
                B_fit[omega_idx],
                PF_conv_fit[omega_idx],
                tau_PF_fit[omega_idx],
            ),
            lw=2,
        )

    ax[0].set_xlim(-0.25, 15.25)
    ax[1].set_xlim(0, 15)
    ax[0].set_xlabel(r"$\tau_\mathrm{max}$", fontsize=36)
    ax[1].set_xlabel(r"$\tau_\mathrm{max}$", fontsize=36)
    ax[0].set_ylabel("Polarization fraction (%)", fontsize=30)
    ax[1].set_ylabel(r"$\omega$", fontsize=36)
    ax[0].set_title(rf"Emergent polarization ($i={angle_deg:.1f}^\circ$)")

    for axis in ax:
        axis.minorticks_on()
        axis.xaxis.set_minor_locator(AutoMinorLocator())
        axis.yaxis.set_minor_locator(AutoMinorLocator())
        axis.tick_params(which="major", direction="in", top=True, right=True, length=8, width=1.2)
        axis.tick_params(which="minor", direction="in", top=True, right=True, length=4, width=1.0)

    h_all, l_all = ax[0].get_legend_handles_labels()
    h_numerical = ax[0].collections[-1]
    h_fitting = ax[0].lines[-1]
    legend1 = ax[0].legend(
        [h_numerical, h_fitting],
        ["Numerical result", f"Polynomial fitting formula (deg={degree})"],
        loc="upper right",
        fontsize=20,
    )
    ax[0].add_artist(legend1)
    ax[0].legend(h_all, l_all, loc="upper left", fontsize=20, bbox_to_anchor=(1.01, 1))

    X, Y = np.meshgrid(data.tau_max_values, data.omega_values)
    error = np.zeros((len(data.omega_values), len(data.tau_max_values)))

    for omega_idx in range(len(data.omega_values)):
        for tau_idx in range(len(data.tau_max_values)):
            model_val = model_func_PF(
                data.tau_max_values[tau_idx],
                A_fit[omega_idx],
                B_fit[omega_idx],
                PF_conv_fit[omega_idx],
                tau_PF_fit[omega_idx],
            )
            data_val = pf_angle[omega_idx, tau_idx]
            error[omega_idx, tau_idx] = np.abs(
                100.0 * (data_val - model_val) / max(np.abs(data_val), EPS)
            )

    mesh = ax[1].pcolormesh(X, Y, error, shading="auto", cmap="viridis", vmin=0, vmax=5)
    cbar = fig.colorbar(mesh, ax=ax[1])
    cbar.set_label("Relative error (%)")

    if savepath is not None:
        fig.savefig(savepath, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)

    return {
        "angle_deg": angle_deg,
        "degree": degree,
        "PF_angle": pf_angle,
        "A_PF": A_PF,
        "B_PF": B_PF,
        "PF_conv": PF_conv,
        "tau_PF": tau_PF,
        "poly_A": poly_A,
        "poly_B": poly_B,
        "poly_PFconv": poly_PFconv,
        "poly_tauPF": poly_tauPF,
        "error": error,
        "data": data,
    }


# =========================
# Evaluate PF from fitted polynomial coefficients
# =========================
def evaluate_PF_from_polynomial(omega, tau_max, poly_A, poly_B, poly_PFconv, poly_tauPF):
    """
    Evaluate the fitted PF model at the requested omega and tau_max.

    Parameters
    ----------
    omega : float or array-like
        Single-scattering albedo. Validated range: 0.0 < omega <= 0.9.
    tau_max : float or array-like
        Maximum optical depth. Must satisfy tau_max >= 0.0.
    poly_A, poly_B, poly_PFconv, poly_tauPF : np.poly1d
        Polynomial functions of omega returned by :func:`fit_and_plot_PF`.

    Returns
    -------
    float or ndarray
        Evaluated polarization fraction.
    """
    omega, tau_max = _validate_evaluation_inputs(omega=omega, tau_max=tau_max)

    A_val = poly_A(omega)
    B_val = poly_B(omega)
    PF_conv_val = poly_PFconv(omega)
    tau_PF_val = poly_tauPF(omega)

    pf_val = model_func_PF(tau_max, A_val, B_val, PF_conv_val, tau_PF_val)

    if np.ndim(pf_val) == 0:
        return float(pf_val)
    return pf_val


def evaluate_PF_from_fit_result(omega, tau_max, fit_result):
    """
    Evaluate the fitted PF model using the dictionary returned by
    :func:`fit_and_plot_PF`.
    """
    required_keys = ["poly_A", "poly_B", "poly_PFconv", "poly_tauPF"]
    missing_keys = [key for key in required_keys if key not in fit_result]
    if missing_keys:
        raise KeyError(f"fit_result is missing required keys: {missing_keys}")

    return evaluate_PF_from_polynomial(
        omega=omega,
        tau_max=tau_max,
        poly_A=fit_result["poly_A"],
        poly_B=fit_result["poly_B"],
        poly_PFconv=fit_result["poly_PFconv"],
        poly_tauPF=fit_result["poly_tauPF"],
    )
