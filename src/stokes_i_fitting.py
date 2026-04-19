"""Fitting utilities for emergent Stokes I."""

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

MIN_VALID_INC_DEG = 0.0
MAX_VALID_INC_DEG = 80.0
MAX_VALID_OMEGA = 0.9


@dataclass(frozen=True)
class StokesIData:
    """Container for the Stokes-I fitting tables."""

    tau_max_values: np.ndarray
    omega_values: np.ndarray
    mu_pos_grid: np.ndarray
    I_surface: np.ndarray
    indir: Path


def _resolve_angle_deg(inc=None, angle_deg=None, *, default=45.0):
    """
    Resolve the public inclination argument.

    Both ``inc`` and ``angle_deg`` are accepted for backward compatibility.
    """
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


def _validate_evaluation_inputs(omega, tau_max, mu):
    """Validate user-supplied evaluation inputs."""
    omega = np.asarray(omega, dtype=float)
    tau_max = np.asarray(tau_max, dtype=float)
    mu = float(mu)

    if np.any((omega < 0.0) | (omega > MAX_VALID_OMEGA)):
        raise ValueError(
            "omega is outside the validated range for this fitting formula. "
            f"Use 0.0 <= omega <= {MAX_VALID_OMEGA}."
        )

    if np.any(tau_max < 0.0):
        raise ValueError("tau_max must satisfy tau_max >= 0.0.")

    if not (0.0 < mu <= 1.0):
        raise ValueError("mu must satisfy 0.0 < mu <= 1.0.")

    return omega, tau_max, mu


# =========================
# Read .inp files
# =========================
def read_single_inp(filepath):
    """
    Read a single .inp file and return
    omega, tau_max, mu_pos, and I_pos.
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
    i_pos = np.asarray(arr[:, 1], dtype=float)

    if len(mu_pos) != n_mu:
        raise ValueError(f"{filepath}: header n_mu and actual row count do not match")

    return float(omega), float(tau_max), mu_pos, i_pos


# =========================
# Load all .inp files into an explicit data bundle
# =========================
def load_I_only_inp(indir=DEFAULT_INDIR_I):
    """
    Read a set of .inp files with headers and reconstruct
    tau_max_values, omega_values, mu_pos_grid, and I_surface.

    Returns
    -------
    StokesIData
        Explicit data bundle that can be passed to other functions instead of
        relying on module-level global variables.
    """
    indir = Path(indir)

    filelist = sorted(indir.glob("*.inp"))
    if not filelist:
        raise FileNotFoundError(f"no inp files found in {indir}")

    records = {}
    tau_set = set()
    omega_set = set()

    for filepath in filelist:
        omega, tau_max, mu_pos, i_pos = read_single_inp(filepath)
        key = (float(omega), float(tau_max))
        if key in records:
            raise ValueError(
                f"Duplicate Stokes-I file found for omega={omega}, "
                f"tau_max={tau_max}: {filepath}"
            )

        records[key] = {
            "filepath": filepath,
            "omega": omega,
            "tau_max": tau_max,
            "mu_pos": mu_pos,
            "I_pos": i_pos,
        }
        tau_set.add(tau_max)
        omega_set.add(omega)

    tau_max_values = np.array(sorted(tau_set), dtype=float)
    omega_values = np.array(sorted(omega_set), dtype=float)

    n_tau = len(tau_max_values)
    n_omega = len(omega_values)

    tau_to_idx = {v: i for i, v in enumerate(tau_max_values)}
    omega_to_idx = {v: i for i, v in enumerate(omega_values)}

    mu_pos_grid = np.full((n_tau,), None, dtype=object)
    i_surface = np.empty((n_omega, n_tau), dtype=object)
    filled = np.zeros((n_omega, n_tau), dtype=bool)

    for rec in records.values():
        i_tau = tau_to_idx[rec["tau_max"]]
        i_omega = omega_to_idx[rec["omega"]]

        if mu_pos_grid[i_tau] is None:
            mu_pos_grid[i_tau] = rec["mu_pos"]
        else:
            same_len = len(mu_pos_grid[i_tau]) == len(rec["mu_pos"])
            same_grid = np.allclose(mu_pos_grid[i_tau], rec["mu_pos"])
            if not (same_len and same_grid):
                raise ValueError(
                    f"Inconsistent mu grid for tau_max={rec['tau_max']} "
                    f"between files. File: {rec['filepath']}"
                )

        i_surface[i_omega, i_tau] = rec["I_pos"]
        filled[i_omega, i_tau] = True

    if not np.all(filled):
        missing = np.argwhere(~filled)
        raise ValueError(f"Some (omega, tau_max) pairs are missing: {missing}")

    return StokesIData(
        tau_max_values=tau_max_values,
        omega_values=omega_values,
        mu_pos_grid=mu_pos_grid,
        I_surface=i_surface,
        indir=indir,
    )


def _coerce_i_surface_inputs(data=None, mu_pos_grid=None, I_surface=None):
    """
    Accept either a ``StokesIData`` bundle or the raw arrays used in the
    original public API.
    """
    if data is not None:
        return data.mu_pos_grid, data.I_surface

    if mu_pos_grid is None or I_surface is None:
        raise TypeError(
            "Pass either data=<StokesIData> or both mu_pos_grid and I_surface."
        )

    return mu_pos_grid, I_surface


# =========================
# Interpolate Stokes I to a requested angle
# =========================
def get_I_at_angle_fast(
    inc=45.0,
    mu_pos_grid=None,
    I_surface=None,
    data=None,
    angle_deg=None,
):
    """
    Interpolate the emergent Stokes I onto the requested viewing angle.

    Parameters
    ----------
    inc : float, optional
        Inclination angle in degrees.
    mu_pos_grid, I_surface : ndarray, optional
        Raw arrays kept for backward compatibility with the original API.
    data : StokesIData, optional
        Explicit data bundle returned by :func:`load_I_only_inp`.
    angle_deg : float or None, optional
        Backward-compatible alias for ``inc``.
    """
    angle_deg = _resolve_angle_deg(inc=inc, angle_deg=angle_deg)
    _validate_fit_inputs(angle_deg=angle_deg, degree=0)
    mu_pos_grid, I_surface = _coerce_i_surface_inputs(
        data=data,
        mu_pos_grid=mu_pos_grid,
        I_surface=I_surface,
    )

    mu = np.cos(np.deg2rad(angle_deg))

    n_omega = I_surface.shape[0]
    n_tau = I_surface.shape[1]
    i_angle = np.zeros((n_omega, n_tau))

    for omega_idx in range(n_omega):
        for tau_idx in range(n_tau):
            mu_grid = mu_pos_grid[tau_idx]
            i_grid = I_surface[omega_idx, tau_idx]

            f = interp1d(mu_grid, i_grid, kind="cubic")
            i_angle[omega_idx, tau_idx] = float(f(mu))

    return i_angle


# =========================
# Stokes I model
# =========================
def stokes_i_model_with_mu(tau_max, A_I, B_I, I_conv, omega_I, mu):
    """Fitting formula for emergent Stokes I with fixed ``mu``."""
    tau_max = np.asarray(tau_max)

    y_left = tau_max / mu * (1.0 - omega_I)
    y_right = (
        I_conv * (1.0 - np.exp(-A_I * tau_max**B_I))
        + 0.04 / mu * (1.0 - omega_I)
        - I_conv * (1.0 - np.exp(-A_I * 0.04**B_I))
    )
    return np.where(tau_max <= 0.04, y_left, y_right)


def make_model_func_fixed_mu(mu):
    """Return a wrapper function with ``mu`` fixed for lmfit."""

    def model_func_fixed_mu(tau_max, A_I, B_I, I_conv, omega_I):
        return stokes_i_model_with_mu(tau_max, A_I, B_I, I_conv, omega_I, mu)

    return model_func_fixed_mu


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
def fit_and_plot_I(
    inc=45.0,
    angle_deg=None,
    degree=4,
    indir=DEFAULT_INDIR_I,
    savepath=None,
    show=True,
    data=None,
):
    """
    Perform the Stokes-I fitting and make the plot.

    Parameters
    ----------
    inc : float, optional
        Inclination angle in degrees. Use keyword arguments when calling this
        function in user-facing examples.
    angle_deg : float or None, optional
        Backward-compatible alias for ``inc``. If given, it overrides ``inc``.
    degree : int, optional
        Polynomial degree for fitting the omega dependence.
    indir : str or Path, optional
        Directory containing .inp files. Ignored when ``data`` is given.
    savepath : str or Path or None, optional
        Output figure path.
    show : bool, optional
        If True, show the figure.
    data : StokesIData or None, optional
        Explicit data bundle returned by :func:`load_I_only_inp`. Passing this
        avoids module-level global state and repeated reloading.
    """
    angle_deg = _resolve_angle_deg(inc=inc, angle_deg=angle_deg)
    _validate_fit_inputs(angle_deg=angle_deg, degree=degree)

    if data is None:
        data = load_I_only_inp(indir=indir)

    mu = np.cos(np.deg2rad(angle_deg))
    i_angle = get_I_at_angle_fast(inc=angle_deg, data=data)

    model_fixed_mu = make_model_func_fixed_mu(mu)
    my_model = lmfit.Model(model_fixed_mu)

    A_I = np.zeros(len(data.omega_values))
    B_I = np.zeros(len(data.omega_values))
    I_conv = np.zeros(len(data.omega_values))
    omega_I = np.zeros(len(data.omega_values))

    for omega_idx in range(len(data.omega_values)):
        ydata = i_angle[omega_idx, :]
        weights = 1.0 / np.maximum(np.abs(ydata), EPS)

        params = my_model.make_params(A_I=1.0, B_I=0.5, I_conv=1.0, omega_I=0.5)
        params["A_I"].set(min=0.0)
        params["B_I"].set(min=0.0)
        params["I_conv"].set(min=0.0)
        params["omega_I"].set(min=0.0, max=1.0)

        results = my_model.fit(
            ydata,
            tau_max=data.tau_max_values,
            params=params,
            weights=weights,
            nan_policy="omit",
            method="least_squares",
            fit_kws={"loss": "linear"},
        )

        A_I[omega_idx] = results.best_values["A_I"]
        B_I[omega_idx] = results.best_values["B_I"]
        I_conv[omega_idx] = results.best_values["I_conv"]
        omega_I[omega_idx] = results.best_values["omega_I"]

    poly_A = np.poly1d(np.polyfit(data.omega_values, A_I, degree))
    poly_B = np.poly1d(np.polyfit(data.omega_values, B_I, degree))
    poly_I = np.poly1d(np.polyfit(data.omega_values, I_conv, degree))
    poly_w = np.poly1d(np.polyfit(data.omega_values, omega_I, degree))

    A_I_fit = poly_A(data.omega_values)
    B_I_fit = poly_B(data.omega_values)
    I_conv_fit = poly_I(data.omega_values)
    omega_I_fit = poly_w(data.omega_values)

    latex_A = poly_to_latex(poly_A)
    latex_B = poly_to_latex(poly_B)
    latex_I = poly_to_latex(poly_I)
    latex_w = poly_to_latex(poly_w)

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
    fig.subplots_adjust(top=0.65, left=0.12, right=0.95, bottom=0.08, hspace=0.32)

    fig.text(0.05, 0.975, r"$\bf{Fitting\ formula\ for\ } I$", ha="left", va="top", fontsize=20)
    fig.text(
        0.05,
        0.940,
        r"$I=(1-\omega_{\mathrm{I}}(\omega,\mu))\frac{\tau_{\max}}{\mu}B"
        r"\qquad (\tau_{\max}<0.04)$",
        ha="left",
        va="top",
        fontsize=18,
    )
    fig.text(
        0.05,
        0.905,
        r"$I=[\,I_{\mathrm{conv}}(\omega,\mu)\left(1-\exp\left(-A_{\mathrm{I}}(\omega,\mu)\tau_{\max}^{B_{\mathrm{I}}(\omega,\mu)}\right)\right)-I_{\mathrm{conv}}(\omega,\mu)\left(1-\exp\left(-A_{\mathrm{I}}(\omega,\mu)0.04^{B_{\mathrm{I}}(\omega,\mu)}\right)\right)+(1-\omega_{\mathrm{I}}(\omega,\mu))\frac{0.04}{\mu}\,]B\qquad (\tau_{\max}>0.04)$",
        ha="left",
        va="top",
        fontsize=18,
    )
    fig.text(0.05, 0.855, rf"$\bf{{Polynomial\ fitting\ formulas\ at\ }} i={angle_deg:.1f}^\circ$", ha="left", va="top", fontsize=20)
    fig.text(0.05, 0.820, rf"$A_{{\mathrm{{I}}}}(\omega) = {latex_A}$", ha="left", va="top", fontsize=18)
    fig.text(0.05, 0.788, rf"$B_{{\mathrm{{I}}}}(\omega) = {latex_B}$", ha="left", va="top", fontsize=18)
    fig.text(0.05, 0.756, rf"$I_{{\mathrm{{conv}}}}(\omega) = {latex_I}$", ha="left", va="top", fontsize=18)
    fig.text(0.05, 0.724, rf"$\omega_{{\mathrm{{I}}}}(\omega) = {latex_w}$", ha="left", va="top", fontsize=18)

    tau_plot = np.arange(0.0, 15.0, 0.01)

    for omega_idx in range(len(data.omega_values)):
        current_omega = data.omega_values[omega_idx]

        ax[0].scatter(
            data.tau_max_values,
            i_angle[omega_idx, :],
            label=fr"$\omega={current_omega}$",
            s=40,
        )
        ax[0].plot(
            tau_plot,
            stokes_i_model_with_mu(
                tau_plot,
                A_I_fit[omega_idx],
                B_I_fit[omega_idx],
                I_conv_fit[omega_idx],
                omega_I_fit[omega_idx],
                mu,
            ),
            lw=2,
        )

    ax[0].set_xlim(-0.25, 15.25)
    ax[0].set_ylim(-0.025, 1.025)
    ax[1].set_xlim(0, 15)
    ax[0].set_xlabel(r"$\tau_\mathrm{max}$", fontsize=36)
    ax[1].set_xlabel(r"$\tau_\mathrm{max}$", fontsize=36)
    ax[0].set_ylabel(r"$I/B$", fontsize=36)
    ax[1].set_ylabel(r"$\omega$", fontsize=36)
    ax[0].set_title(rf"Emergent Stokes $I$ ($i={angle_deg:.1f}^\circ$)")

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
        loc="lower right",
        fontsize=20,
    )
    ax[0].add_artist(legend1)
    ax[0].legend(h_all, l_all, loc="upper left", fontsize=20, bbox_to_anchor=(1.01, 1))

    X, Y = np.meshgrid(data.tau_max_values, data.omega_values)
    error = np.zeros((len(data.omega_values), len(data.tau_max_values)))

    for omega_idx in range(len(data.omega_values)):
        for tau_idx in range(len(data.tau_max_values)):
            model_val = stokes_i_model_with_mu(
                data.tau_max_values[tau_idx],
                A_I_fit[omega_idx],
                B_I_fit[omega_idx],
                I_conv_fit[omega_idx],
                omega_I_fit[omega_idx],
                mu,
            )
            data_val = i_angle[omega_idx, tau_idx]
            error[omega_idx, tau_idx] = np.abs(
                100.0 * (data_val - model_val) / max(np.abs(data_val), EPS)
            )

    mesh = ax[1].pcolormesh(X, Y, error, shading="auto", cmap="viridis", vmin=0, vmax=3)
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
        "mu": mu,
        "I_angle": i_angle,
        "A_I": A_I,
        "B_I": B_I,
        "I_conv": I_conv,
        "omega_I": omega_I,
        "poly_A": poly_A,
        "poly_B": poly_B,
        "poly_I": poly_I,
        "poly_w": poly_w,
        "error": error,
        "data": data,
    }


# Backward-compatible aliases
read_single_light_inp = read_single_inp
load_I_only_light_inp = load_I_only_inp


# =========================
# Evaluate Stokes I from fitted polynomial coefficients
# =========================
def evaluate_I_from_polynomial(omega, tau_max, mu, poly_A, poly_B, poly_I, poly_w):
    """
    Evaluate the fitted Stokes-I model at the requested omega and tau_max.

    Parameters
    ----------
    omega : float or array-like
        Single-scattering albedo. Validated range: 0.0 <= omega <= 0.9.
    tau_max : float or array-like
        Maximum optical depth. Must satisfy tau_max >= 0.0.
    mu : float
        Cosine of the viewing angle.
    poly_A, poly_B, poly_I, poly_w : np.poly1d
        Polynomial functions of omega returned by :func:`fit_and_plot_I`.

    Returns
    -------
    float or ndarray
        Evaluated Stokes-I value.
    """
    omega, tau_max, mu = _validate_evaluation_inputs(omega=omega, tau_max=tau_max, mu=mu)

    A_val = poly_A(omega)
    B_val = poly_B(omega)
    I_conv_val = poly_I(omega)
    omega_I_val = poly_w(omega)

    i_val = stokes_i_model_with_mu(tau_max, A_val, B_val, I_conv_val, omega_I_val, mu)

    if np.ndim(i_val) == 0:
        return float(i_val)
    return i_val


def evaluate_I_from_fit_result(omega, tau_max, fit_result):
    """
    Evaluate the fitted Stokes-I value using the dictionary returned by
    :func:`fit_and_plot_I`.
    """
    required_keys = ["mu", "poly_A", "poly_B", "poly_I", "poly_w"]
    missing_keys = [key for key in required_keys if key not in fit_result]
    if missing_keys:
        raise KeyError(f"fit_result is missing required keys: {missing_keys}")

    return evaluate_I_from_polynomial(
        omega=omega,
        tau_max=tau_max,
        mu=fit_result["mu"],
        poly_A=fit_result["poly_A"],
        poly_B=fit_result["poly_B"],
        poly_I=fit_result["poly_I"],
        poly_w=fit_result["poly_w"],
    )
