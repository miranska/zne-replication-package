import numpy as np
import pytest

from bounded_methods import (
    bounded_exp_extrapolation,
    bounded_polyexp_extrapolation,
    bounded_polynomial_extrapolation,
)


class FailedResult:
    def __init__(self, message: str = "forced failure") -> None:
        self.success = False
        self.message = message


def _failing_minimize(*args, **kwargs):
    return FailedResult()


@pytest.mark.parametrize(
    "x,y,order,match",
    [
        ([0.0, 1.0], [0.0], 1, "same number of rows"),
        ([0.0], [0.0], 1, "At least two data points"),
        ([0.0, np.nan], [0.0, 0.5], 1, "must be finite"),
        ([0.0, 1.0], [0.0, np.inf], 1, "must be finite"),
        ([0.0, 1.0], [0.0, 0.5], -1, "order must be non-negative"),
        ([0.0, 1.0], [0.0, 0.5], 2, "Need more than order=2 points"),
    ],
)
def test_bounded_polynomial_validation_errors(x, y, order, match):
    with pytest.raises(ValueError, match=match):
        bounded_polynomial_extrapolation(x, y, order=order)


def test_bounded_polynomial_linear_fit_returns_expected_intercept():
    x = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    y = 0.3 + 0.2 * x

    result = bounded_polynomial_extrapolation(x, y, order=1)

    assert result == pytest.approx(0.3, abs=1e-6)


def test_bounded_polynomial_clips_when_trend_intercept_outside_bounds():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = np.array([1.2, 1.1, 1.05, 1.01])

    result = bounded_polynomial_extrapolation(x, y, order=1)

    assert result == pytest.approx(1.0, abs=1e-6)


def test_bounded_polynomial_raises_runtime_error_when_optimizer_fails(monkeypatch):
    monkeypatch.setattr("bounded_methods.minimize", _failing_minimize)

    with pytest.raises(
        RuntimeError, match="Bounded polynomial fit failed: forced failure"
    ):
        bounded_polynomial_extrapolation([0.0, 1.0], [0.2, 0.1], order=1)


@pytest.mark.parametrize(
    "x,y,match",
    [
        ([0.0, 1.0], [0.2], "same number of rows"),
        ([0.0], [0.2], "At least two data points"),
        ([0.0, np.nan], [0.2, 0.1], "must be finite"),
        ([0.0, 1.0], [0.2, np.inf], "must be finite"),
    ],
)
def test_bounded_exp_validation_errors(x, y, match):
    with pytest.raises(ValueError, match=match):
        bounded_exp_extrapolation(x, y)


@pytest.mark.parametrize(
    "asymptote,match",
    [
        (np.nan, "asymptote must be finite or None"),
        (1.2, "asymptote must lie in \\[-1, 1\\]"),
    ],
)
def test_bounded_exp_invalid_asymptote(asymptote, match):
    with pytest.raises(ValueError, match=match):
        bounded_exp_extrapolation([0.0, 1.0], [0.3, 0.1], asymptote=asymptote)


def test_bounded_exp_with_fixed_asymptote_recovers_zero_noise_value():
    x = np.array([0.0, 1.0, 2.0, 3.0])
    asymptote = -0.2
    z0 = 0.7
    c = 0.8
    y = asymptote + (z0 - asymptote) * np.exp(-c * x)

    result = bounded_exp_extrapolation(x, y, asymptote=asymptote)

    assert result == pytest.approx(z0, abs=1e-4)


def test_bounded_exp_with_fitted_asymptote_returns_bounded_result():
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    y = np.array([0.85, 0.55, 0.30, 0.10, -0.02])

    result = bounded_exp_extrapolation(x, y, asymptote=None)

    assert -1.0 <= result <= 1.0
    assert result == pytest.approx(0.85, abs=0.1)


@pytest.mark.parametrize("asymptote", [None, 0.0])
def test_bounded_exp_raises_runtime_error_when_optimizer_fails(monkeypatch, asymptote):
    monkeypatch.setattr("bounded_methods.minimize", _failing_minimize)

    with pytest.raises(
        RuntimeError, match="Bounded exponential fit failed: forced failure"
    ):
        bounded_exp_extrapolation([0.0, 1.0], [0.4, 0.2], asymptote=asymptote)


@pytest.mark.parametrize(
    "x,y,order,asymptote,match",
    [
        ([0.0, 1.0], [0.2], 1, None, "same number of rows"),
        ([0.0], [0.2], 1, None, "At least two data points"),
        ([0.0, np.nan], [0.2, 0.1], 1, None, "must be finite"),
        ([0.0, 1.0], [0.2, 0.1], -1, None, "order must be non-negative"),
        ([0.0, 1.0], [0.2, 0.1], 2, None, "Need more than order=2 points"),
        (
            [0.0, 1.0],
            [0.2, 0.1],
            1,
            np.nan,
            "asymptote must be finite and lie in \\[-1, 1\\], or None",
        ),
        (
            [0.0, 1.0],
            [0.2, 0.1],
            1,
            2.0,
            "asymptote must be finite and lie in \\[-1, 1\\], or None",
        ),
    ],
)
def test_bounded_polyexp_validation_errors(x, y, order, asymptote, match):
    with pytest.raises(ValueError, match=match):
        bounded_polyexp_extrapolation(x, y, order=order, asymptote=asymptote)


def test_bounded_polyexp_with_fixed_asymptote_recovers_zero_noise_value():
    x = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    asymptote = -0.4
    z0 = 0.6
    coeff = -0.8
    y = asymptote + (z0 - asymptote) * np.exp(coeff * x)

    result = bounded_polyexp_extrapolation(x, y, order=1, asymptote=asymptote)

    assert result == pytest.approx(z0, abs=1e-4)


def test_bounded_polyexp_without_asymptote_recovers_zero_noise_value():
    x = np.array([0.0, 0.7, 1.4, 2.1, 2.8, 3.5])
    asymptote = -0.1
    z0 = 0.9
    coeff = -0.6
    y = asymptote + (z0 - asymptote) * np.exp(coeff * x)

    result = bounded_polyexp_extrapolation(x, y, order=1, asymptote=None)

    assert -1.0 <= result <= 1.0
    assert result == pytest.approx(z0, abs=5e-2)


def test_bounded_polyexp_returns_bounded_output_for_extreme_data():
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.array([1.3, 1.2, 1.15, 1.1])

    result = bounded_polyexp_extrapolation(x, y, order=1)

    assert -1.0 <= result <= 1.0


def test_bounded_polyexp_raises_runtime_error_when_optimizer_fails(monkeypatch):
    monkeypatch.setattr("bounded_methods.minimize", _failing_minimize)

    with pytest.raises(
        RuntimeError, match="Bounded PolyExp fit failed: forced failure"
    ):
        bounded_polyexp_extrapolation([0.0, 1.0], [0.6, 0.3], order=1)
