"""Bounded methods for extrapolation."""

from __future__ import annotations


import numpy as np
import pandas as pd
from scipy.optimize import minimize


def bounded_polynomial_extrapolation(
    x: pd.Series | np.ndarray | list,
    y: pd.Series | np.ndarray | list,
    order: int = 1,
) -> float:
    """Polynomial extrapolation with bounded physically valid zero-noise prediction.

    Model:
        y(x) = theta_0 + theta_1 * x + ... + theta_d * x^d

    The zero-noise limit is y(0) = theta_0 (the intercept). The fit is constrained
    so that theta_0 lies in [-1, 1], ensuring the extrapolated value is valid for
    ±1-valued observables.

    :param x: Series containing the scale factors.
    :param y: Series containing the expectation values.
    :param order: Order of the polynomial.
    :return: Zero-noise limit y(0).
    :raises ValueError: If inputs are invalid or too few points.
    :raises RuntimeError: If constrained optimization fails.
    """
    x = np.asarray(x, dtype=float)
    y_vals = np.asarray(y, dtype=float)

    if x.size != y_vals.size:
        raise ValueError("x and y must have the same number of rows.")
    if x.size < 2:
        raise ValueError("At least two data points are required.")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y_vals)):
        raise ValueError("scale_factor and expectation values must be finite.")
    if order < 0:
        raise ValueError("order must be non-negative.")
    if x.size <= order:
        raise ValueError(
            f"Need more than order={order} points for a unique fit, got {x.size}."
        )

    # Design matrix: columns 1, x, x^2, ..., x^d
    A = np.column_stack([x**k for k in range(order + 1)])

    def objective(theta: np.ndarray) -> float:
        pred = A @ theta
        return float(np.sum((y_vals - pred) ** 2))

    # Initial guess from unconstrained OLS
    theta_init, _, _, _ = np.linalg.lstsq(A, y_vals, rcond=None)

    # Bounds: intercept theta_0 in [-1, 1], higher coefficients unbounded
    bounds = [(-1.0, 1.0)] + [(None, None)] * order

    result = minimize(
        objective,
        x0=theta_init,
        bounds=bounds,
        method="L-BFGS-B",
    )

    if not result.success:
        raise RuntimeError(f"Bounded polynomial fit failed: {result.message}")

    zero_noise_value = result.x[0]
    return float(np.clip(zero_noise_value, -1.0, 1.0))


def bounded_exp_extrapolation(
    x: pd.Series | np.ndarray | list,
    y: pd.Series | np.ndarray | list,
    asymptote: float | None = None,
) -> float:
    """Exponential extrapolation with physical bounds.

    Model:
        y(x) = a + (z0 - a) * exp(-c * x),   c > 0

    This reparameterization makes y(0)=z0 explicit, so z0 can be bounded
    in [-1, 1] directly.

    :param x: Series containing the scale factors.
    :param y: Series containing the expectation values.
    :param asymptote: Infinite-noise limit. If None, it is estimated from data.
    :return: Zero-noise limit y(0).
    """
    x = np.asarray(x, dtype=float)
    y_vals = np.asarray(y, dtype=float)

    if x.size != y_vals.size:
        raise ValueError("x and y must have the same number of rows.")
    if x.size < 2:
        raise ValueError("At least two data points are required.")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y_vals)):
        raise ValueError("scale_factor and expectation values must be finite.")

    c_min = 1e-12

    if asymptote is not None:
        if not np.isfinite(asymptote):
            raise ValueError("asymptote must be finite or None.")
        if asymptote < -1.0 or asymptote > 1.0:
            raise ValueError("asymptote must lie in [-1, 1].")

        def objective(params: np.ndarray) -> float:
            z0, c = params
            pred = asymptote + (z0 - asymptote) * np.exp(-c * x)
            return float(np.sum((y_vals - pred) ** 2))

        z00 = float(np.clip(y_vals[0], -1.0, 1.0))
        c0 = 0.1
        result = minimize(
            objective,
            x0=np.array([z00, c0]),
            bounds=[(-1.0, 1.0), (c_min, None)],
            method="L-BFGS-B",
        )
        if not result.success:
            raise RuntimeError(f"Bounded exponential fit failed: {result.message}")
        zero_noise_value = result.x[0]
    else:

        def objective(params: np.ndarray) -> float:
            a, z0, c = params
            pred = a + (z0 - a) * np.exp(-c * x)
            return float(np.sum((y_vals - pred) ** 2))

        a0 = float(np.clip(y_vals[-1], -1.0, 1.0))
        z00 = float(np.clip(y_vals[0], -1.0, 1.0))
        c0 = 0.1
        result = minimize(
            objective,
            x0=np.array([a0, z00, c0]),
            bounds=[(-1.0, 1.0), (-1.0, 1.0), (c_min, None)],
            method="L-BFGS-B",
        )
        if not result.success:
            raise RuntimeError(f"Bounded exponential fit failed: {result.message}")
        zero_noise_value = result.x[1]

    return float(np.clip(zero_noise_value, -1.0, 1.0))


def bounded_polyexp_extrapolation(
    x: pd.Series | np.ndarray | list,
    y: pd.Series | np.ndarray | list,
    order: int = 1,
    asymptote: float | None = None,
) -> float:
    """Constrained polynomial-exponential extrapolation with bounded zero-noise value.

    Model:
        y(x) = a + (z0 - a) * exp(r(x)),
    where:
        r(x) = c1 * x + c2 * x^2 + ... + c_order * x^order

    This parameterization forces y(0) = z0 explicitly, so constraining z0 in [-1, 1]
    directly enforces a physically valid zero-noise prediction.

    :param x: Series containing the scale factors.
    :param y: Series containing expectation values.
    :param order: Polynomial order in the exponent.
    :param asymptote: Infinite-noise limit. If None, fit in [-1, 1].
    :return: Zero-noise limit y(0).
    :raises ValueError: If inputs are invalid.
    :raises RuntimeError: If constrained optimization fails.
    """
    x = np.asarray(x, dtype=float)
    y_vals = np.asarray(y, dtype=float)

    if x.size != y_vals.size:
        raise ValueError("x and y must have the same number of rows.")
    if x.size < 2:
        raise ValueError("At least two data points are required.")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y_vals)):
        raise ValueError("scale_factor and expectation values must be finite.")
    if order < 0:
        raise ValueError("order must be non-negative.")
    if x.size <= order:
        raise ValueError(
            f"Need more than order={order} points for a stable fit, got {x.size}."
        )
    if asymptote is not None and (
        not np.isfinite(asymptote) or not (-1.0 <= asymptote <= 1.0)
    ):
        raise ValueError("asymptote must be finite and lie in [-1, 1], or None.")

    def _poly_no_intercept(x_: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
        if coeffs.size == 0:
            return np.zeros_like(x_, dtype=float)
        powers = np.column_stack([x_ ** (k + 1) for k in range(coeffs.size)])
        return powers @ coeffs

    def _predict(params: np.ndarray) -> np.ndarray:
        if asymptote is None:
            a = params[0]
            z0 = params[1]
            coeffs = params[2:]
        else:
            a = float(asymptote)
            z0 = params[0]
            coeffs = params[1:]
        return a + (z0 - a) * np.exp(_poly_no_intercept(x, coeffs))

    def objective(params: np.ndarray) -> float:
        residuals = y_vals - _predict(params)
        return float(np.sum(residuals**2))

    if asymptote is None:
        a0 = float(np.clip(y_vals[-1], -1.0, 1.0))
        z00 = float(np.clip(y_vals[0], -1.0, 1.0))
        x0 = np.concatenate([np.array([a0, z00]), np.zeros(order, dtype=float)])
        bounds = [(-1.0, 1.0), (-1.0, 1.0)] + [(None, None)] * order
    else:
        z00 = float(np.clip(y_vals[0], -1.0, 1.0))
        x0 = np.concatenate([np.array([z00]), np.zeros(order, dtype=float)])
        bounds = [(-1.0, 1.0)] + [(None, None)] * order

    result = minimize(objective, x0=x0, bounds=bounds, method="L-BFGS-B")
    if not result.success:
        raise RuntimeError(f"Bounded PolyExp fit failed: {result.message}")

    zero_noise_value = result.x[1] if asymptote is None else result.x[0]
    return float(np.clip(zero_noise_value, -1.0, 1.0))
