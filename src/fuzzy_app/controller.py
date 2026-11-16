"""Координатор GUI: связывает независимые библиотеки приложения."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from fuzzy_core.models import FuzzyDataset, GaussianFuzzyNumber
from fuzzy_core.level_sets import level_interval
from fuzzy_core.normal_generator import (
    GenerationConfig,
    config_to_dict,
    generate_dataset,
)
from fuzzy_clustering import ClusterResult, cluster_dataset
from fuzzy_storage import (
    load_dataset_tsv,
    save_cluster_result,
    save_dataset_tsv,
    save_generation_artifacts,
    next_dataset_stem,
)


class FuzzyLabController:
    """Управляет набором нечётких величин и кластеризацией."""

    def __init__(self, root_directory: str | Path = ".") -> None:
        self.root = Path(root_directory)

        # Создаём и очищаем журнал при запуске приложения.
        self.log_path = self.root / "logs" / "fuzzy_lab.txt"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        started_at = datetime.now().astimezone().isoformat(timespec="seconds")
        self.log_path.write_text(
            "Журнал событий Fuzzy Lab\n"
            f"Запуск: {started_at}\n"
            + "=" * 72 + "\n",
            encoding="utf-8",
        )
        self.dataset = FuzzyDataset()
        self.h = 0.7
        self.last_generation_metadata: dict | None = None
        self.last_generation_paths: tuple[Path, Path, Path] | None = None
        self.last_cluster_result: ClusterResult | None = None

        self._cluster_parameters = {"k": 4, "max_k": 10}


    @staticmethod
    def _item_details(item: GaussianFuzzyNumber) -> dict[str, Any]:
        return {
            "id": int(item.id),
            "x0": float(item.x0),
            "sigma": float(item.sigma),
        }

    def _record_event(
        self,
        event_type: str,
        summary: str,
        details: dict[str, Any] | None = None,
    ) -> int:
        """Записывает необходимые события в текстовый журнал."""

        event_type = event_type.strip().upper()
        details = dict(details or {})

        count_events = {
            "GENERATE_DATASET",
            "ADD_NUMBER",
            "DELETE_NUMBER",
            "LOAD_DATASET",
        }

        if event_type not in count_events | {
            "CHANGE_ALPHA",
            "CLUSTER_DATASET",
        }:
            return 0

        timestamp = datetime.now().astimezone().isoformat(timespec="seconds")

        total_count = len(self.dataset)
        included_count = sum(
            level_interval(item, self.h) is not None
            for item in self.dataset.items
        )

        lines = [f"[{timestamp}]"]

        if event_type in count_events:
            lines.extend([
                "ИЗМЕНЕНИЕ КОЛИЧЕСТВА ВЕЛИЧИН",
                f"Операция: {summary}",
                f"Всего величин: {total_count}",
                f"Подходят уровню α={self.h:g}: {included_count}",
            ])

        elif event_type == "CHANGE_ALPHA":
            previous_alpha = details.get(
                "previous_alpha",
                details.get(
                    "previous_h",
                    details.get("old_alpha", details.get("old_h")),
                ),
            )

            lines.extend([
                "ИЗМЕНЕНИЕ УРОВНЯ ALPHA",
                f"Операция: {summary}",
            ])

            if previous_alpha is not None:
                lines.append(f"Предыдущий уровень α: {previous_alpha}")

            lines.extend([
                f"Новый уровень α: {self.h:g}",
                f"Всего величин: {total_count}",
                f"Подходят новому уровню α: {included_count}",
            ])

        elif event_type == "CLUSTER_DATASET":
            cluster_count = details.get("cluster_count")

            if cluster_count is None:
                clusters = details.get("clusters")
                if isinstance(clusters, (dict, list, tuple, set)):
                    cluster_count = len(clusters)

            if cluster_count is None:
                cluster_count = details.get("selected_k")

            if cluster_count is None:
                cluster_count = "не определено"

            alpha = details.get("alpha", self.h)

            lines.extend([
                "ЗАВЕРШЕНИЕ КЛАСТЕРИЗАЦИИ",
                f"Операция: {summary}",
                f"Количество кластеров: {cluster_count}",
                f"Уровень α: {alpha}",
            ])

        with self.log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(
                "\n".join(lines)
                + "\n"
                + "-" * 72
                + "\n"
            )

        return 0

    def record_action(
        self,
        event_type: str,
        summary: str,
        details: dict[str, Any] | None = None,
    ) -> int:
        """Публичная точка записи событий, возникающих в GUI."""
        return self._record_event(event_type, summary, details)

    def generate(self, config: GenerationConfig) -> FuzzyDataset:
        dataset, metadata = generate_dataset(config)
        self.dataset = dataset
        self.last_generation_metadata = metadata
        self.last_generation_paths = None
        self.last_cluster_result = None

        generation_rows = {
            int(row["id"]): row
            for row in metadata.get("generation_rows", [])
        }
        generated_items = []
        for item in self.dataset.items:
            row = generation_rows.get(item.id, {})
            generated_items.append(
                {
                    **self._item_details(item),
                    "omega_n": float(row["omega_n"]),
                    "eta": float(row["eta"]),
                }
            )

        details = {
            "dataset_name": self.dataset.name,
            "count": len(self.dataset),
            "generation_config": config_to_dict(config),
            "generation_metadata": {
                key: value
                for key, value in metadata.items()
                if key != "generation_rows"
            },
            "generated_items": generated_items,
        }
        self._record_event(
            "GENERATE_DATASET",
            f"Сгенерировано {len(self.dataset)} нечётких величин",
            details,
        )
        return self.dataset

    def add(self, x0: float, sigma: float) -> GaussianFuzzyNumber:
        item = GaussianFuzzyNumber(self.dataset.next_id(), x0, sigma)
        self.dataset.add(item)
        self.last_cluster_result = None
        self._record_event(
            "ADD_NUMBER",
            f"Добавлена нечёткая величина A{item.id}",
            {
                "dataset_name": self.dataset.name,
                "item": self._item_details(item),
                "count_after": len(self.dataset),
            },
        )
        return item

    def edit(
        self,
        item_id: int,
        x0: float,
        sigma: float,
    ) -> GaussianFuzzyNumber:
        old_item = self.dataset.get(item_id)
        replacement = GaussianFuzzyNumber(item_id, x0, sigma)
        for index, item in enumerate(self.dataset.items):
            if item.id == item_id:
                self.dataset.items[index] = replacement
                self.last_cluster_result = None
                self._record_event(
                    "EDIT_NUMBER",
                    f"Отредактирована нечёткая величина A{item_id}",
                    {
                        "dataset_name": self.dataset.name,
                        "item_id": item_id,
                        "before": self._item_details(old_item),
                        "after": self._item_details(replacement),
                        "count_after": len(self.dataset),
                    },
                )
                return replacement
        raise KeyError(f"Элемент id={item_id} не найден")

    def delete(self, item_id: int) -> None:
        removed_item = self.dataset.get(item_id)
        self.dataset.remove(item_id)
        self.last_cluster_result = None
        self._record_event(
            "DELETE_NUMBER",
            f"Удалена нечёткая величина A{item_id}",
            {
                "dataset_name": self.dataset.name,
                "deleted_item": self._item_details(removed_item),
                "count_after": len(self.dataset),
            },
        )

    def set_level(self, h: float) -> None:
        if h <= 0:
            raise ValueError("h должен быть положительным")
        if self.h != h:
            previous_h = self.h
            self.h = h
            self.last_cluster_result = None
            included_count = sum(
                level_interval(item, h) is not None
                for item in self.dataset.items
            )
            self._record_event(
                "CHANGE_ALPHA",
                f"Изменён уровень α: {previous_h:g} → {h:g}",
                {
                    "previous_alpha": previous_h,
                    "new_alpha": h,
                    "dataset_count": len(self.dataset),
                    "included_count": included_count,
                    "excluded_count": len(self.dataset) - included_count,
                },
            )

    def load(self, path: str | Path) -> FuzzyDataset:
        loaded_dataset = load_dataset_tsv(path)
        self.dataset = loaded_dataset
        self.last_generation_metadata = None
        self.last_generation_paths = None
        self.last_cluster_result = None
        self._record_event(
            "LOAD_DATASET",
            f"Загружен набор из {len(self.dataset)} нечётких величин",
            {
                "path": str(Path(path).resolve()),
                "dataset_name": self.dataset.name,
                "count": len(self.dataset),
                "items": [self._item_details(item) for item in self.dataset.items],
            },
        )
        return self.dataset

    def save(self, path: str | Path) -> Path:
        result = save_dataset_tsv(self.dataset, path)
        self._record_event(
            "SAVE_DATASET",
            f"Сохранён набор из {len(self.dataset)} нечётких величин",
            {
                "path": str(result.resolve()),
                "dataset_name": self.dataset.name,
                "count": len(self.dataset),
            },
        )
        return result

    def save_generation(self) -> tuple[Path, Path, Path]:
        if self.last_generation_metadata is None:
            raise ValueError("Текущий набор не был создан генератором")
        if (
            self.last_generation_paths is not None
            and all(path.exists() for path in self.last_generation_paths)
        ):
            return self.last_generation_paths

        datasets_directory = self.root / "data" / "datasets"
        stem = next_dataset_stem(datasets_directory)
        result = save_generation_artifacts(
            self.dataset,
            self.last_generation_metadata,
            datasets_directory,
            stem=stem,
        )
        self.last_generation_paths = result
        self._record_event(
            "SAVE_GENERATION_ARTIFACTS",
            f"Сохранены файлы генерации {stem}",
            {
                "dataset_path": str(result[0].resolve()),
                "metadata_path": str(result[1].resolve()),
                "trace_path": str(result[2].resolve()),
                "count": len(self.dataset),
            },
        )
        return result

    def set_cluster_parameter(self, parameter: str, value: int) -> None:
        """Отдельно фиксирует изменение K или максимального K."""
        if parameter not in self._cluster_parameters:
            raise ValueError("Параметр должен быть 'k' или 'max_k'")
        new_value = int(value)
        if new_value < 1:
            raise ValueError("Количество кластеров должно быть положительным")

        previous_value = self._cluster_parameters[parameter]
        if previous_value == new_value:
            return

        self._cluster_parameters[parameter] = new_value
        parameter_title = "K" if parameter == "k" else "Максимум K"
        self._record_event(
            "CHANGE_CLUSTER_COUNT",
            f"Изменён параметр {parameter_title}: {previous_value} → {new_value}",
            {
                "parameter": parameter,
                "previous_value": int(previous_value),
                "new_value": new_value,
                "alpha": float(self.h),
                "dataset_count": int(len(self.dataset)),
            },
        )

    def cluster(
        self,
        mode: str,
        k: int = 4,
        max_k: int = 10,
    ) -> ClusterResult:
        result = cluster_dataset(
            self.dataset,
            self.h,
            mode=mode,
            k=k,
            max_k=max_k,
        )
        self.last_cluster_result = result

        items_by_id = {
            item.id: item
            for item in [*result.included_items, *result.excluded_items]
        }
        clusters_details = {
            str(cluster_id): {
                "item_ids": list(item_ids),
                "medoid_id": result.medoids[cluster_id],
                "items": [
                    self._item_details(items_by_id[item_id])
                    for item_id in item_ids
                ],
            }
            for cluster_id, item_ids in result.clusters.items()
        }

        details = {
            "dataset_name": self.dataset.name,
            "mode": mode,
            "requested_k": k if mode == "fixed" else None,
            "max_k": max_k if mode == "auto" else None,
            "alpha": result.h,
            "algorithm": result.algorithm,
            "selected_k": result.selected_k,
            "included_count": len(result.included_items),
            "excluded_count": len(result.excluded_items),
            "included_items": [
                self._item_details(item)
                for item in result.included_items
            ],
            "excluded_items": [
                self._item_details(item)
                for item in result.excluded_items
            ],
            "clusters": clusters_details,
            "medoids": {
                str(cluster_id): medoid_id
                for cluster_id, medoid_id in result.medoids.items()
            },
            "silhouette_score": (
                float(result.silhouette_score)
                if result.silhouette_score is not None
                else None
            ),
            "silhouette_by_k": {
                str(candidate_k): float(score)
                for candidate_k, score in result.silhouette_by_k.items()
            },
            "distance_metric": "Hausdorff distance between alpha-level intervals",
            "linkage": "average",
        }
        self._record_event(
            "CLUSTER_DATASET",
            f"Кластеризация завершена: K={result.selected_k}, "
            f"включено {len(result.included_items)}, "
            f"исключено {len(result.excluded_items)}",
            details,
        )
        return result

    def save_clustering(self) -> Path:
        if self.last_cluster_result is None:
            raise ValueError("Сначала выполните кластеризацию")
        path = save_cluster_result(
            self.dataset,
            self.last_cluster_result,
            self.root / "data" / "cluster_results",
        )
        self._record_event(
            "SAVE_CLUSTER_RESULT",
            f"Сохранён результат кластеризации: {path.name}",
            {"path": str(path.resolve()), "run_directory": path.name},
        )
        return path
