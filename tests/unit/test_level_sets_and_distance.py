import math
import pytest

from fuzzy_core.models import GaussianFuzzyNumber
from fuzzy_core.level_sets import level_interval
from fuzzy_core.distance import hausdorff_interval_distance
from fuzzy_core.level_sets import LevelInterval


def test_level_interval_boundaries_satisfy_density_equation():
    from fuzzy_core.gaussian import gaussian_density

    item = GaussianFuzzyNumber(1, 2.0, 0.3)
    h = 0.7
    interval = level_interval(item, h)
    assert interval is not None
    assert gaussian_density(interval.left, item.x0, item.sigma) == pytest.approx(h)
    assert gaussian_density(interval.right, item.x0, item.sigma) == pytest.approx(h)
    assert interval.left < item.x0 < interval.right


def test_level_interval_tangent_case():
    from fuzzy_core.gaussian import maximum_density

    item = GaussianFuzzyNumber(1, -3.0, 0.5)
    interval = level_interval(item, maximum_density(item.sigma))
    assert interval is not None
    assert interval.left == pytest.approx(item.x0)
    assert interval.right == pytest.approx(item.x0)


def test_level_interval_empty_when_h_above_peak():
    item = GaussianFuzzyNumber(1, 0.0, 1.0)
    assert level_interval(item, 0.7) is None


def test_level_must_be_positive():
    item = GaussianFuzzyNumber(1, 0.0, 0.2)
    with pytest.raises(ValueError):
        level_interval(item, 0.0)


def test_hausdorff_distance_for_intervals():
    first = LevelInterval(9.0, 11.0)
    second = LevelInterval(10.0, 13.0)
    assert hausdorff_interval_distance(first, second) == pytest.approx(2.0)
    assert hausdorff_interval_distance(first, second) == pytest.approx(
        hausdorff_interval_distance(second, first)
    )
