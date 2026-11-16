"""Пример вызова библиотек Fuzzy Lab из сторонней программы без GUI."""

from fuzzy_core.normal_generator import GenerationConfig, generate_dataset
from fuzzy_core.level_sets import level_interval
from fuzzy_clustering import cluster_dataset
from fuzzy_storage import save_dataset_tsv


def main() -> None:
    # Создаём 120 гауссовских величин с центрами по формулам задания.
    config = GenerationConfig(
        count=120,
        center_mean=0.0,
        center_std=1.0,
        n=8,
        approximation="formula_1",  # первая поправка со знаком минус
        seed=2026,
        sigma_mode="uniform",
        sigma_min=0.15,
        sigma_max=0.45,
    )
    dataset, _metadata = generate_dataset(config)

    # Независимая библиотека fuzzy_clustering получает математическую модель,
    # а не GUI-компоненты.
    result = cluster_dataset(dataset, h=0.7, mode="auto", max_k=8)
    print(f"Всего объектов: {len(dataset)}")
    print(f"Включено при h=0.7: {len(result.included_items)}")
    print(f"Исключено при h=0.7: {len(result.excluded_items)}")
    print(f"Выбрано кластеров: {result.selected_k}")
    print(f"Средний силуэт: {result.silhouette_score:.6f}")
    print("Кластеры:", result.clusters)

    for item in dataset.items[:3]:
        print(f"A{item.id}: x0={item.x0:.6f}, sigma={item.sigma:.6f}, I_h={level_interval(item, 0.7)}")

    save_dataset_tsv(dataset, "external_dataset.tsv")
    print("Исходный набор сохранён в external_dataset.tsv")


if __name__ == "__main__":
    main()
