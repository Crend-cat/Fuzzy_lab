"""Функции для работы с исходной плотностью нормального распределения."""

from __future__ import annotations

import math
import numpy as np

from .models import GaussianFuzzyNumber


SQRT_2PI = math.sqrt(2.0 * math.pi)


def gaussian_density(x: float | np.ndarray, x0: float, sigma: float):
    """Вычислить плотность без дополнительной нормировки."""
    if not math.isfinite(x0):
        raise ValueError("x0 должен быть конечным")
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma должна быть конечной и положительной")
    values = np.asarray(x, dtype=float)
    result = np.exp(-((values - x0) ** 2) / (2.0 * sigma**2)) / (sigma * SQRT_2PI)
    if np.isscalar(x):
        return float(result)
    return result


def density_for_number(x: float | np.ndarray, number: GaussianFuzzyNumber):
    """Вычислить плотность для объекта модели."""
    return gaussian_density(x, number.x0, number.sigma)


def maximum_density(sigma: float) -> float:
    """Максимум плотности, достигаемый в точке x0."""
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma должна быть конечной и положительной")
    return 1.0 / (sigma * SQRT_2PI)


def standard_normal_density(x: float | np.ndarray):
    """Эталонная стандартная нормальная плотность N(0, 1)."""
    return gaussian_density(x, 0.0, 1.0)
