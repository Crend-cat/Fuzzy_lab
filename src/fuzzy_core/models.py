"""Модели данных математического ядра."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Iterable


@dataclass(frozen=True, slots=True)
class GaussianFuzzyNumber:
    """Одна гауссовская нечёткая величина.

    Время создания намеренно не хранится. Объект определяется только
    идентификатором и параметрами исходной функции плотности.
    """

    id: int
    x0: float
    sigma: float

    def __post_init__(self) -> None:
        if not isinstance(self.id, int) or self.id <= 0:
            raise ValueError("id должен быть положительным целым числом")
        if not math.isfinite(self.x0):
            raise ValueError("x0 должен быть конечным числом")
        if not math.isfinite(self.sigma) or self.sigma <= 0:
            raise ValueError("sigma должна быть конечным положительным числом")


@dataclass(slots=True)
class FuzzyDataset:
    """Набор нечётких величин с уникальными идентификаторами."""

    items: list[GaussianFuzzyNumber] = field(default_factory=list)
    name: str = "Набор нечётких величин"

    def __post_init__(self) -> None:
        self.replace_items(self.items)

    def replace_items(self, items: Iterable[GaussianFuzzyNumber]) -> None:
        values = list(items)
        identifiers = [item.id for item in values]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Идентификаторы элементов набора должны быть уникальными")
        self.items = values

    def add(self, item: GaussianFuzzyNumber) -> None:
        if any(existing.id == item.id for existing in self.items):
            raise ValueError(f"Элемент с id={item.id} уже существует")
        self.items.append(item)

    def remove(self, item_id: int) -> GaussianFuzzyNumber:
        for index, item in enumerate(self.items):
            if item.id == item_id:
                return self.items.pop(index)
        raise KeyError(f"Элемент id={item_id} не найден")

    def get(self, item_id: int) -> GaussianFuzzyNumber:
        for item in self.items:
            if item.id == item_id:
                return item
        raise KeyError(f"Элемент id={item_id} не найден")

    def next_id(self) -> int:
        return max((item.id for item in self.items), default=0) + 1

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self):
        return iter(self.items)
