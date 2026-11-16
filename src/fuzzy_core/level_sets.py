"""Подмножества уровня h и их граничные точки."""

from __future__ import annotations

from dataclasses import dataclass
import math

from .gaussian import maximum_density
from .models import GaussianFuzzyNumber


@dataclass(frozen=True, slots=True)
class LevelInterval:
    """Непустое множество уровня, заданное замкнутым интервалом."""

    left: float
    right: float

    def __post_init__(self) -> None:
        if not (math.isfinite(self.left) and math.isfinite(self.right)):
            raise ValueError("Границы интервала должны быть конечными")
        if self.left > self.right:
            raise ValueError("Левая граница не может быть больше правой")


def level_interval(number: GaussianFuzzyNumber, h: float) -> LevelInterval | None:
    """Решить f(x; x0, sigma) = h.

    None означает, что горизонтальная прямая выше максимума плотности и
    множество {x: f(x) >= h} пусто. При касании возвращается [x0, x0].
    """
    if not math.isfinite(h) or h <= 0:
        raise ValueError("Уровень h должен быть конечным положительным числом")

    peak = maximum_density(number.sigma)
    tolerance = 1e-12 * max(1.0, peak, h)
    if h > peak + tolerance:
        return None
    if math.isclose(h, peak, rel_tol=1e-12, abs_tol=1e-14):
        return LevelInterval(number.x0, number.x0)

    radicand = -2.0 * math.log(h * number.sigma * math.sqrt(2.0 * math.pi))
    # Небольшая отрицательная погрешность округления у касания исправляется нулём.
    radicand = max(0.0, radicand)
    offset = number.sigma * math.sqrt(radicand)
    return LevelInterval(number.x0 - offset, number.x0 + offset)


def included_at_level(number: GaussianFuzzyNumber, h: float) -> bool:
    """Проверить, имеет ли функция непустое множество уровня h."""
    return level_interval(number, h) is not None
