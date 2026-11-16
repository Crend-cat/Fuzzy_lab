"""Структуры результатов кластеризации."""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from fuzzy_core.models import GaussianFuzzyNumber


@dataclass(slots=True)
class ClusterResult:
    h: float
    mode: str
    algorithm: str
    included_items: list[GaussianFuzzyNumber]
    excluded_items: list[GaussianFuzzyNumber]
    labels: dict[int, int]
    clusters: dict[int, list[int]]
    medoids: dict[int, int]
    distance_matrix: np.ndarray
    linkage_matrix: np.ndarray | None
    selected_k: int
    silhouette_score: float | None = None
    silhouette_by_k: dict[int, float] = field(default_factory=dict)
