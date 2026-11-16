"""Метрики между непустыми множествами уровня."""

from __future__ import annotations

from .level_sets import LevelInterval


def hausdorff_interval_distance(first: LevelInterval, second: LevelInterval) -> float:
    """Расстояние Хаусдорфа между двумя замкнутыми интервалами на R.

    Для I=[l1,r1], J=[l2,r2] оно равно max(|l1-l2|, |r1-r2|).
    """
    return max(abs(first.left - second.left), abs(first.right - second.right))
