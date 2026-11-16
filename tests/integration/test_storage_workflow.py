from fuzzy_core.models import FuzzyDataset, GaussianFuzzyNumber
from fuzzy_core.normal_generator import GenerationConfig, generate_dataset
from fuzzy_clustering import cluster_dataset
from fuzzy_storage.tsv_repository import (
    save_dataset_tsv,
    load_dataset_tsv,
    save_generation_artifacts,
    save_cluster_result,
)


def test_dataset_round_trip(tmp_path):
    dataset = FuzzyDataset([GaussianFuzzyNumber(1, 1.2, 0.3), GaussianFuzzyNumber(2, -2.0, 0.4)])
    path = save_dataset_tsv(dataset, tmp_path / "dataset.tsv")
    loaded = load_dataset_tsv(path)
    assert loaded.items == dataset.items


def test_generation_artifacts_are_saved(tmp_path):
    dataset, metadata = generate_dataset(GenerationConfig(count=100, seed=2))
    paths = save_generation_artifacts(dataset, metadata, tmp_path, stem="dataset_0001")
    assert all(path.exists() for path in paths)


def test_cluster_result_is_saved_in_new_run_folder(tmp_path):
    dataset, _ = generate_dataset(GenerationConfig(count=100, seed=3))
    result = cluster_dataset(dataset, h=0.7, mode="fixed", k=3)
    run_dir = save_cluster_result(dataset, result, tmp_path)
    assert (run_dir / "clusters.tsv").exists()
    assert (run_dir / "metadata.json").exists()
