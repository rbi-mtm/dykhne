"""
Landau-Zener transition probability via the Dykhne formula.

For a grid of LZ parameters (a, b) and fitting half-widths T, this script:
  1. Computes the two energy levels E0, E1 from the LZ model.
  2. Forms deltaE = E1 - E0.
  3. Fits deltaE on [-T, T] using the AAA rational approximant (FitAAA).
  4. Integrates from the real axis to the nearest complex zero of deltaE
     along a vertical path (ClosestDirect).
  5. Computes the numerical transition probability P = exp(-2 * Im(X)),
     where X is the contour integral.
  6. Compares with the exact LZ result P_exact = exp(-pi * b^2 / a).

The LZ energy levels are:
    E0(t) = -sqrt(b^2 + a^2 * t^2)
    E1(t) = +sqrt(b^2 + a^2 * t^2)

deltaE has branch-point zeros at t = ±i*b/a.  The nearest zero in the
upper half-plane is t0 = i*b/a, and the exact action gives
    Im(X) = pi * b^2 / (2*a)
leading to P_exact = exp(-pi * b^2 / a).
"""

import numpy as np

from dykhne.models.lz import LZ
from dykhne.fit.fit_aaa import FitAAA
from dykhne.integrate.closest_direct import ClosestDirect

# ---------------------------------------------------------------------------
# Global time grid (wide enough to capture the full level structure)
# ---------------------------------------------------------------------------
T_GLOBAL = 10.0
STEP = 0.1

# ---------------------------------------------------------------------------
# Parameter grid
# ---------------------------------------------------------------------------
A_VALUES = [0.5, 1.0, 2.0]
B_VALUES = [0.5, 1.0, 2.0]
T_VALUES = [1.0, 2.0, 3.0, 5.0, 7.0, 10.0]   # half-width of the AAA fitting window


def compute_delta_e(a, b, t):
    """Return deltaE = E1 - E0 for the LZ model at parameter values a, b."""
    levels = LZ(a=a, b=b).energy_levels(t)
    return levels['1'] - levels['0']


def run(a, b, T, t_global):
    """
    Run the full pipeline for a single (a, b, T) combination.

    Parameters
    ----------
    a, b : float
        LZ model parameters.
    T : float
        Half-width of the fitting window [-T, T].
    t_global : np.ndarray
        Dense time grid used to sample the energy levels.

    Returns
    -------
    P : float
        Numerical transition probability.
    P_exact : float
        Exact Landau-Zener transition probability.
    root : complex
        Complex zero used as the integration endpoint.
    integral : complex
        Value of the contour integral.
    """
    delta_e = compute_delta_e(a, b, t_global)

    mask = np.abs(t_global) <= T
    t_fit = t_global[mask]
    e_fit = delta_e[mask]

    fit = FitAAA()
    fit.fit(t_fit, e_fit)

    integrator = ClosestDirect(fit, upper_half_plane=True)
    integral, root = integrator.integrate()

    P = np.exp(-2.0 * np.imag(integral))
    P_exact = np.exp(-np.pi * b**2 / a)

    return P, P_exact, root, integral


def print_header():
    print(
        f"{'a':>5}  {'b':>5}  {'T':>5}  "
        f"{'P_numerical':>14}  {'P_exact':>14}  "
        f"{'rel. error':>12}  "
        f"{'root (Im)':>12}  {'Im root exact':>14}  {'Im(X)':>12}"
    )
    print("-" * 112)


def print_row(a, b, T, P, P_exact, root, integral):
    rel_err = abs(P - P_exact) / P_exact if P_exact > 0 else float('nan')
    im_root_exact = b / a
    print(
        f"{a:>5.2f}  {b:>5.2f}  {T:>5.1f}  "
        f"{P:>14.8f}  {P_exact:>14.8f}  "
        f"{rel_err:>12.2e}  "
        f"{root.imag:>12.6f}  {im_root_exact:>14.6f}  {np.imag(integral):>12.6f}"
    )


if __name__ == '__main__':

    t_global = np.arange(-T_GLOBAL, T_GLOBAL + STEP, STEP)

    print_header()

    for a in A_VALUES:
        for b in B_VALUES:
            for T in T_VALUES:
                try:
                    P, P_exact, root, integral = run(a, b, T, t_global)
                    print_row(a, b, T, P, P_exact, root, integral)
                except Exception as e:
                    print(
                        f"  a={a:.2f}  b={b:.2f}  T={T:.1f}  "
                        f"FAILED: {e}"
                    )
