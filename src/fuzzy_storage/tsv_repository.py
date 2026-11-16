"""TSV/JSON-хранение исходных наборов и результатов анализа."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from fuzzy_core.models import FuzzyDataset, GaussianFuzzyNumber
from fuzzy_core.level_sets import level_interval
from fuzzy_clustering.results import ClusterResult


def save_dataset_tsv(dataset: FuzzyDataset, path: str | Path) -> Path:
    """Сохранить набор: одна строка TSV — одна нечёткая величина."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, delimiter="\t", lineterminator="\n")
        writer.writerow(["id", "x0", "sigma"])
        for item in dataset.items:
            writer.writerow([item.id, format(item.x0, ".17g"), format(item.sigma, ".17g")])
    temporary.replace(destination)
    return destination


def load_dataset_tsv(path: str | Path) -> FuzzyDataset:
    """Загрузить набор TSV и проверить все записи моделью."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    values: list[GaussianFuzzyNumber] = []
    with source.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file, delimiter="\t")
        required = {"id", "x0", "sigma"}
        if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
            raise ValueError("TSV должен содержать столбцы id, x0, sigma")
        for line_number, row in enumerate(reader, start=2):
            try:
                values.append(
                    GaussianFuzzyNumber(
                        id=int(row["id"]),
                        x0=float(row["x0"]),
                        sigma=float(row["sigma"]),
                    )
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Некорректная запись в строке {line_number}: {exc}") from exc
    return FuzzyDataset(items=values, name=source.stem)


def save_json(data: dict[str, Any], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2, allow_nan=False)
    temporary.replace(destination)
    return destination


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError("JSON с метаданными должен содержать объект")
    return data



def next_dataset_stem(root: str | Path, prefix: str = "dataset_") -> str:
    """Выбрать свободное имя dataset_NNNN, не перезаписывая предыдущие наборы."""
    base = Path(root)
    base.mkdir(parents=True, exist_ok=True)
    numbers = []
    for path in base.glob(f"{prefix}*.tsv"):
        suffix = path.stem[len(prefix):]
        if suffix.isdigit():
            numbers.append(int(suffix))
    return f"{prefix}{max(numbers, default=0) + 1:04d}"

def save_generation_artifacts(
    dataset: FuzzyDataset,
    metadata: dict[str, Any],
    directory: str | Path,
    stem: str = "dataset_001",
) -> tuple[Path, Path, Path]:
    """Сохранить исходный TSV, JSON-параметры и подробный лог генерации."""
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    dataset_path = save_dataset_tsv(dataset, root / f"{stem}.tsv")

    generation_rows = metadata.get("generation_rows", [])
    clean_metadata = {key: value for key, value in metadata.items() if key != "generation_rows"}
    metadata_path = save_json(clean_metadata, root / f"{stem}_metadata.json")

    trace_path = root / f"{stem}_generation_trace.tsv"
    with trace_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, delimiter="\t", lineterminator="\n")
        writer.writerow(["id", "omega_n", "eta", "x0", "sigma"])
        for row in generation_rows:
            writer.writerow(
                [
                    row["id"],
                    format(row["omega_n"], ".17g"),
                    format(row["eta"], ".17g"),
                    format(row["x0"], ".17g"),
                    format(row["sigma"], ".17g"),
                ]
            )
    return dataset_path, metadata_path, trace_path


def next_run_directory(root: str | Path) -> Path:
    """Создать следующий каталог run_NNNN, не перезаписывая старые результаты."""
    base = Path(root)
    base.mkdir(parents=True, exist_ok=True)
    existing_numbers = []
    for child in base.glob("run_*"):
        if child.is_dir():
            try:
                existing_numbers.append(int(child.name.split("_", 1)[1]))
            except (IndexError, ValueError):
                continue
    number = max(existing_numbers, default=0) + 1
    result = base / f"run_{number:04d}"
    result.mkdir(parents=True, exist_ok=False)
    return result


def save_cluster_result(
    dataset: FuzzyDataset,
    result: ClusterResult,
    root: str | Path,
) -> Path:
    """Сохранить разметку всех элементов, включая исключённые выбранным h."""
    run_dir = next_run_directory(root)
    included_by_id = {item.id: item for item in result.included_items}
    excluded_ids = {item.id for item in result.excluded_items}

    with (run_dir / "clusters.tsv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, delimiter="\t", lineterminator="\n")
        writer.writerow(["id", "x0", "sigma", "included", "cluster_id", "x_left", "x_right"])
        for item in dataset.items:
            if item.id in included_by_id:
                interval = level_interval(item, result.h)
                writer.writerow(
                    [
                        item.id,
                        format(item.x0, ".17g"),
                        format(item.sigma, ".17g"),
                        "true",
                        result.labels[item.id],
                        format(interval.left, ".17g"),
                        format(interval.right, ".17g"),
                    ]
                )
            elif item.id in excluded_ids:
                writer.writerow([item.id, format(item.x0, ".17g"), format(item.sigma, ".17g"), "false", "", "", ""])

    metadata = {
        "h": result.h,
        "mode": result.mode,
        "algorithm": result.algorithm,
        "selected_k": result.selected_k,
        "silhouette_score": result.silhouette_score,
        "silhouette_by_k": {str(key): value for key, value in result.silhouette_by_k.items()},
        "clusters": {str(key): value for key, value in result.clusters.items()},
        "medoids": {str(key): value for key, value in result.medoids.items()},
        "included_count": len(result.included_items),
        "excluded_count": len(result.excluded_items),
        "distance_metric": "Hausdorff distance between h-level intervals",
        "linkage": "average",
    }
    save_json(metadata, run_dir / "metadata.json")
    return run_dir
