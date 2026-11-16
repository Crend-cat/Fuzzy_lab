"""Публичная процедура иерархической кластеризации."""

from __future__ import annotations

import numpy as np

from fuzzy_core.models import FuzzyDataset
from .distance_matrix import build_distance_matrix
from .linkage import build_average_linkage, cut_hierarchy
from .results import ClusterResult
from .silhouette import mean_silhouette


def _select_medoids(items, labels: np.ndarray, distances: np.ndarray) -> dict[int, int]:
    medoids: dict[int, int] = {}
    for cluster_id in sorted(set(int(value) for value in labels)):
        indices = np.flatnonzero(labels == cluster_id)
        if len(indices) == 1:
            medoids[cluster_id] = items[int(indices[0])].id
            continue
        submatrix = distances[np.ix_(indices, indices)]
        sums = np.sum(submatrix, axis=1)
        best_local = int(np.argmin(sums))
        medoids[cluster_id] = items[int(indices[best_local])].id
    return medoids


def cluster_dataset(
    dataset: FuzzyDataset,
    h: float = 0.7,
    mode: str = "fixed",
    k: int = 4,
    max_k: int = 10,
) -> ClusterResult:
    """Кластеризовать непустые множества уровня h.

    mode='fixed': заданное количество k.
    mode='auto': выбор k по максимуму среднего коэффициента силуэта.
    """
    if mode not in ("fixed", "auto"):
        raise ValueError("mode должен быть 'fixed' или 'auto'")

    prepared = build_distance_matrix(dataset, h)
    items = prepared.included_items
    distances = prepared.matrix
    count = len(items)
    if count == 0:
        raise ValueError("При выбранном h нет величин с непустым множеством уровня")

    linkage_matrix = build_average_linkage(distances)
    scores_by_k: dict[int, float] = {}

    if mode == "fixed":
        selected_k = k
        labels = cut_hierarchy(linkage_matrix, selected_k, count)
        selected_score = None
    else:
        if count < 3:
            raise ValueError("Автоматический выбор по силуэту требует минимум 3 включённых объекта")
        if max_k < 2:
            raise ValueError("max_k должен быть не меньше 2")
        upper = min(max_k, count - 1)
        if upper < 2:
            raise ValueError("Недостаточно объектов для автоматического выбора числа кластеров")

        # Перебираем K по возрастанию. При равном силуэте останется меньшее K.
        best_k: int | None = None
        best_score = -float("inf")
        for candidate_k in range(2, upper + 1):
            candidate_labels = cut_hierarchy(linkage_matrix, candidate_k, count)
            score = mean_silhouette(distances, candidate_labels)
            scores_by_k[candidate_k] = score
            if score > best_score + 1e-12:
                best_score = score
                best_k = candidate_k
        if best_k is None:
            raise RuntimeError("Не удалось подобрать количество кластеров")
        selected_k = best_k
        selected_score = best_score
        labels = cut_hierarchy(linkage_matrix, selected_k, count)

    labels_by_id: dict[int, int] = {}
    clusters: dict[int, list[int]] = {}
    for item, label in zip(items, labels, strict=True):
        cluster_id = int(label)
        labels_by_id[item.id] = cluster_id
        clusters.setdefault(cluster_id, []).append(item.id)
    medoids = _select_medoids(items, labels, distances)

    return ClusterResult(
        h=h,
        mode=mode,
        algorithm="Agglomerative Average Linkage",
        included_items=items,
        excluded_items=prepared.excluded_items,
        labels=labels_by_id,
        clusters=clusters,
        medoids=medoids,
        distance_matrix=distances,
        linkage_matrix=linkage_matrix,
        selected_k=selected_k,
        silhouette_score=selected_score,
        silhouette_by_k=scores_by_k,
    )
