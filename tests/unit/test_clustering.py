import pytest

from fuzzy_core.models import FuzzyDataset, GaussianFuzzyNumber
from fuzzy_clustering import cluster_dataset, build_distance_matrix


def make_dataset():
    values = []
    centers = [-10, -9.8, -10.2, 0, 0.2, -0.2, 10, 10.2, 9.8]
    widths = [0.2, 0.22, 0.18, 0.2, 0.22, 0.18, 0.2, 0.22, 0.18]
    for index, (center, width) in enumerate(zip(centers, widths), start=1):
        values.append(GaussianFuzzyNumber(index, center, width))
    return FuzzyDataset(values)


def test_distance_matrix_is_symmetric_and_zero_diagonal():
    prepared = build_distance_matrix(make_dataset(), 0.7)
    assert prepared.matrix.shape == (9, 9)
    assert (prepared.matrix.diagonal() == 0).all()
    assert (prepared.matrix == prepared.matrix.T).all()


def test_fixed_clustering_assigns_every_included_item():
    result = cluster_dataset(make_dataset(), h=0.7, mode="fixed", k=3)
    assert result.selected_k == 3
    assert len(result.labels) == 9
    assert len(result.clusters) == 3
    assert sum(len(items) for items in result.clusters.values()) == 9


def test_auto_clustering_selects_valid_k():
    result = cluster_dataset(make_dataset(), h=0.7, mode="auto", max_k=5)
    assert 2 <= result.selected_k <= 5
    assert result.silhouette_score is not None
    assert result.silhouette_score == pytest.approx(result.silhouette_by_k[result.selected_k])


def test_objects_above_threshold_are_excluded_not_deleted():
    dataset = FuzzyDataset([
        GaussianFuzzyNumber(1, 0.0, 0.2),
        GaussianFuzzyNumber(2, 1.0, 2.0),
        GaussianFuzzyNumber(3, 2.0, 0.2),
    ])
    result = cluster_dataset(dataset, h=0.7, mode="fixed", k=2)
    assert len(result.included_items) == 2
    assert [item.id for item in result.excluded_items] == [2]
    assert len(dataset) == 3
