"""Построение матрицы расстояний между активными нечёткими величинами."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from fuzzy_core.models import GaussianFuzzyNumber, FuzzyDataset
from fuzzy_core.level_sets import LevelInterval, level_interval
from fuzzy_core.distance import hausdorff_interval_distance


@dataclass(slots=True)
class DistanceMatrix:
    included_items: list[GaussianFuzzyNumber]
    excluded_items: list[GaussianFuzzyNumber]
    intervals: list[LevelInterval]
    matrix: np.ndarray


def build_distance_matrix(dataset: FuzzyDataset, h: float) -> DistanceMatrix:
    """Сформировать матрицу Хаусдорфа для всех непустых h-интервалов."""
    included: list[GaussianFuzzyNumber] = []
    excluded: list[GaussianFuzzyNumber] = []
    intervals: list[LevelInterval] = []

    for item in dataset.items:
        interval = level_interval(item, h)
        if interval is None:
            excluded.append(item)
        else:
            included.append(item)
            intervals.append(interval)

    count = len(included)
    matrix = np.zeros((count, count), dtype=float)
    for i in range(count):
        for j in range(i + 1, count):
            distance = hausdorff_interval_distance(intervals[i], intervals[j])
            matrix[i, j] = distance
            matrix[j, i] = distance

    return DistanceMatrix(included, excluded, intervals, matrix)
