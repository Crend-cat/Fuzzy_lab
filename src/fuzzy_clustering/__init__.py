"""Независимая библиотека кластеризации."""

from .agglomerative import cluster_dataset
from .results import ClusterResult
from .distance_matrix import build_distance_matrix, DistanceMatrix
from .silhouette import mean_silhouette

__all__ = ["cluster_dataset", "ClusterResult", "build_distance_matrix", "DistanceMatrix", "mean_silhouette"]
