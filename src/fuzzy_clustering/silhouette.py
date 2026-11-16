"""Расчёт силуэта по предварительно вычисленной матрице расстояний."""

from __future__ import annotations

import numpy as np


def mean_silhouette(distance_matrix: np.ndarray, labels: np.ndarray) -> float:
    """Средний коэффициент силуэта для матрицы попарных расстояний.

    Для singleton-кластера вклад объекта принимается равным нулю — стандартная
    практическая договорённость для силуэта одиночных кластеров.
    """
    distances = np.asarray(distance_matrix, dtype=float)
    labels = np.asarray(labels)
    count = distances.shape[0]
    if distances.shape != (count, count) or labels.shape != (count,):
        raise ValueError("Размеры матрицы расстояний и массива labels не согласованы")
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2 or len(unique_labels) >= count:
        raise ValueError("Для силуэта требуется 2 <= число кластеров < число объектов")

    cluster_indices = {label: np.flatnonzero(labels == label) for label in unique_labels}
    scores = np.zeros(count, dtype=float)

    for index in range(count):
        own_label = labels[index]
        own_indices = cluster_indices[own_label]
        if len(own_indices) <= 1:
            scores[index] = 0.0
            continue

        own_others = own_indices[own_indices != index]
        a_value = float(np.mean(distances[index, own_others]))

        b_value = float("inf")
        for other_label, other_indices in cluster_indices.items():
            if other_label == own_label:
                continue
            mean_distance = float(np.mean(distances[index, other_indices]))
            b_value = min(b_value, mean_distance)

        denominator = max(a_value, b_value)
        scores[index] = 0.0 if denominator == 0.0 else (b_value - a_value) / denominator

    return float(np.mean(scores))
