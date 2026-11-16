import pytest

from fuzzy_core.normal_generator import (
    GenerationConfig,
    approximate_standard_normal,
    generate_dataset,
    generate_omega_n,
)
import numpy as np


def test_first_approximation_uses_minus_sign():
    omega = 1.25
    n = 8
    expected = omega - (omega**3 - 3 * omega) / (20 * n)
    assert approximate_standard_normal(omega, n, "formula_1") == pytest.approx(expected)


def test_second_approximation_formula():
    omega = -0.75
    n = 4
    expected = omega - (41 / (13440 * n**2)) * (omega**5 - 10 * omega**3 + 15 * omega)
    assert approximate_standard_normal(omega, n, "formula_2") == pytest.approx(expected)


def test_generation_is_reproducible_for_same_seed():
    config = GenerationConfig(count=100, seed=123)
    first, first_metadata = generate_dataset(config)
    second, second_metadata = generate_dataset(config)
    assert first.items == second.items
    assert first_metadata["generation_rows"] == second_metadata["generation_rows"]


def test_uniform_widths_are_inside_configured_bounds():
    config = GenerationConfig(count=100, seed=7, sigma_min=0.2, sigma_max=0.4)
    dataset, _ = generate_dataset(config)
    assert all(0.2 <= item.sigma <= 0.4 for item in dataset)


def test_fixed_width_mode_uses_same_sigma_for_all():
    config = GenerationConfig(count=100, seed=7, sigma_mode="fixed", fixed_sigma=0.31)
    dataset, _ = generate_dataset(config)
    assert {item.sigma for item in dataset} == {0.31}


def test_default_automatic_widths_cross_initial_h_level():
    from fuzzy_core.gaussian import maximum_density

    dataset, _ = generate_dataset(GenerationConfig(count=100, seed=15))
    assert all(maximum_density(item.sigma) > 0.7 for item in dataset)


def test_formula_n_constraints():
    with pytest.raises(ValueError):
        GenerationConfig(count=100, n=4, approximation="formula_1")
    with pytest.raises(ValueError):
        GenerationConfig(count=100, n=1, approximation="formula_2")


def test_generation_count_constraints():
    with pytest.raises(ValueError):
        GenerationConfig(count=99)
