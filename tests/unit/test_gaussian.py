import math

import numpy as np
import pytest

from fuzzy_core.gaussian import gaussian_density, maximum_density, standard_normal_density
from fuzzy_core.models import GaussianFuzzyNumber


def test_density_peak_matches_formula():
    sigma = 0.4
    x0 = 2.5
    assert gaussian_density(x0, x0, sigma) == pytest.approx(1 / (sigma * math.sqrt(2 * math.pi)))
    assert gaussian_density(x0, x0, sigma) == pytest.approx(maximum_density(sigma))


def test_density_is_symmetric():
    x0, sigma, offset = 3.0, 0.7, 1.25
    assert gaussian_density(x0 - offset, x0, sigma) == pytest.approx(
        gaussian_density(x0 + offset, x0, sigma)
    )


def test_density_integrates_to_one_without_extra_normalization():
    sigma, x0 = 0.6, -1.2
    x = np.linspace(x0 - 10 * sigma, x0 + 10 * sigma, 100_001)
    y = gaussian_density(x, x0, sigma)
    area = float(np.sum((y[:-1] + y[1:]) * 0.5 * np.diff(x)))
    assert area == pytest.approx(1.0, rel=1e-7)


def test_standard_normal_is_peak_at_zero():
    assert standard_normal_density(0.0) == pytest.approx(1 / math.sqrt(2 * math.pi))


def test_sigma_must_be_positive():
    with pytest.raises(ValueError):
        GaussianFuzzyNumber(1, 0.0, 0.0)
