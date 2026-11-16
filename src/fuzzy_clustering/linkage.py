"""Иерархическое объединение с методом средней связи."""

from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import linkage, cut_tree
from scipy.spatial.distance import squareform


def build_average_linkage(distance_matrix: np.ndarray) -> np.ndarray | None:
    """Построить linkage matrix по готовой матрице расстояний.

    scipy.cluster.hierarchy.linkage(method='average') выполняет average linkage.
    Возвращается None для менее чем двух объектов.
    """
    matrix = np.asarray(distance_matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Матрица расстояний должна быть квадратной")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Матрица расстояний содержит нечисловые значения")
    if matrix.shape[0] < 2:
        return None
    if not np.allclose(matrix, matrix.T, atol=1e-10):
        raise ValueError("Матрица расстояний должна быть симметричной")
    if not np.allclose(np.diag(matrix), 0.0, atol=1e-10):
        raise ValueError("Диагональ матрицы расстояний должна быть нулевой")
    if np.any(matrix < -1e-12):
        raise ValueError("Расстояния не могут быть отрицательными")

    condensed = squareform(matrix, checks=True)
    return linkage(condensed, method="average")


def cut_hierarchy(linkage_matrix: np.ndarray | None, count: int, object_count: int) -> np.ndarray:
    """Разрезать дендрограмму на заданное число групп.

    cut_tree используется вместо fcluster(maxclust), так как maxclust при
    одинаковых расстояниях может вернуть меньше запрошенного числа групп.
    """
    if not 1 <= count <= object_count:
        raise ValueError(f"Количество кластеров должно быть от 1 до {object_count}")
    if object_count == 1:
        return np.array([1], dtype=int)
    if count == object_count:
        return np.arange(1, object_count + 1, dtype=int)
    if count == 1:
        return np.ones(object_count, dtype=int)
    if linkage_matrix is None:
        raise ValueError("Для разреза дерева недостаточно объектов")

    raw = cut_tree(linkage_matrix, n_clusters=[count]).reshape(-1)
    # Нормируем метки к последовательности 1..K.
    mapping = {old: new for new, old in enumerate(dict.fromkeys(raw.tolist()), start=1)}
    return np.array([mapping[value] for value in raw], dtype=int)
