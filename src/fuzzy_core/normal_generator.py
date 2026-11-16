"""Генерация центров по аппроксимациям из задания и ширин функций."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Literal

import numpy as np

from .models import FuzzyDataset, GaussianFuzzyNumber

Approximation = Literal["formula_1", "formula_2"]
SigmaMode = Literal["uniform", "fixed"]


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    """Параметры генерации набора.

    center_mean / center_std задают преобразование x0_i = center_mean +
    center_std * eta_i. Они не являются индивидуальной шириной sigma_i
    каждой создаваемой нечёткой величины.
    """

    count: int = 150
    center_mean: float = 0.0
    center_std: float = 1.0
    n: int = 8
    approximation: Approximation = "formula_1"
    seed: int | None = 42
    sigma_mode: SigmaMode = "uniform"
    sigma_min: float = 0.15
    sigma_max: float = 0.45
    fixed_sigma: float = 0.3

    def __post_init__(self) -> None:
        if not 100 <= self.count <= 300:
            raise ValueError("Количество генерируемых величин должно быть от 100 до 300")
        if not math.isfinite(self.center_mean):
            raise ValueError("center_mean должен быть конечным")
        if not math.isfinite(self.center_std) or self.center_std <= 0:
            raise ValueError("center_std должен быть конечным положительным числом")
        if self.approximation not in ("formula_1", "formula_2"):
            raise ValueError("Неизвестная аппроксимация")
        minimum_n = 5 if self.approximation == "formula_1" else 2
        if not isinstance(self.n, int) or self.n < minimum_n:
            raise ValueError(f"Для {self.approximation} требуется n >= {minimum_n}")
        if self.sigma_mode not in ("uniform", "fixed"):
            raise ValueError("sigma_mode должен быть 'uniform' или 'fixed'")
        if self.sigma_mode == "uniform":
            if not (math.isfinite(self.sigma_min) and math.isfinite(self.sigma_max)):
                raise ValueError("Границы автоматической генерации sigma должны быть конечными")
            if self.sigma_min <= 0 or self.sigma_max <= self.sigma_min:
                raise ValueError("Требуется 0 < sigma_min < sigma_max")
        else:
            if not math.isfinite(self.fixed_sigma) or self.fixed_sigma <= 0:
                raise ValueError("Единая sigma должна быть конечной и положительной")


def generate_omega_n(n: int, rng: np.random.Generator) -> float:
    """Сформировать omega_n из n независимых U(0,1) по формуле задания."""
    if not isinstance(n, int) or n < 1:
        raise ValueError("n должно быть положительным целым числом")
    uniform_values = rng.uniform(0.0, 1.0, size=n)
    return math.sqrt(12.0 / n) * (float(np.sum(uniform_values)) - n / 2.0)


def approximate_standard_normal(
    omega_n: float,
    n: int,
    approximation: Approximation,
) -> float:
    """Вычислить eta. Знаки и коэффициенты воспроизводят формулы задания.

    Обратите внимание: в первой формуле оставлен знак МИНУС — ровно так,
    как указано в предоставленном пользователем варианте задания.
    """
    if approximation == "formula_1":
        if n < 5:
            raise ValueError("Для первой аппроксимации требуется n >= 5")
        return omega_n - (omega_n**3 - 3.0 * omega_n) / (20.0 * n)
    if approximation == "formula_2":
        if n < 2:
            raise ValueError("Для второй аппроксимации требуется n >= 2")
        correction = (41.0 / (13440.0 * n**2)) * (
            omega_n**5 - 10.0 * omega_n**3 + 15.0 * omega_n
        )
        return omega_n - correction
    raise ValueError(f"Неизвестная аппроксимация: {approximation}")


def generate_one_center(
    center_mean: float,
    center_std: float,
    n: int,
    approximation: Approximation,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    """Вернуть (omega_n, eta, x0_i) для одной генерируемой величины."""
    omega_n = generate_omega_n(n, rng)
    eta = approximate_standard_normal(omega_n, n, approximation)
    center = center_mean + center_std * eta
    return omega_n, eta, center


def generate_dataset(config: GenerationConfig) -> tuple[FuzzyDataset, dict]:
    """Создать набор и метаданные генерации.

    Автоматический закон ширин:
        sigma_i ~ Uniform(sigma_min, sigma_max), независимо для каждого i.

    Обоснование: задание не определяет отдельный вероятностный закон для
    sigma_i. На выбранном конечном интервале равномерный закон не выделяет
    предпочтительные значения (это распределение максимальной дифференциальной
    энтропии среди непрерывных законов с таким фиксированным носителем).
    Нижняя граница строго положительна. Значения по умолчанию 0.15..0.45
    выбраны так, чтобы при h=0.7 максимум каждой генерируемой плотности был
    выше уровня h. Границы можно менять; это проектные настройки, а не
    константы, предписанные преподавателем.
    """
    rng = np.random.default_rng(config.seed)
    items: list[GaussianFuzzyNumber] = []
    generation_rows: list[dict] = []

    for item_id in range(1, config.count + 1):
        omega_n, eta, center = generate_one_center(
            config.center_mean,
            config.center_std,
            config.n,
            config.approximation,
            rng,
        )

        if config.sigma_mode == "uniform":
            # Ширина генерируется независимо от центра на заданном интервале.
            fuzzy_sigma = float(rng.uniform(config.sigma_min, config.sigma_max))
        else:
            # В фиксированном режиме одна ширина используется для всего набора.
            fuzzy_sigma = float(config.fixed_sigma)

        item = GaussianFuzzyNumber(id=item_id, x0=float(center), sigma=fuzzy_sigma)
        items.append(item)
        generation_rows.append(
            {
                "id": item_id,
                "omega_n": omega_n,
                "eta": eta,
                "x0": float(center),
                "sigma": fuzzy_sigma,
            }
        )

    dataset = FuzzyDataset(items=items, name="Сгенерированный набор")
    metadata = {
        "count": config.count,
        "center_mean": config.center_mean,
        "center_std": config.center_std,
        "n": config.n,
        "approximation": config.approximation,
        "formula_1_sign": "minus",
        "seed": config.seed,
        "sigma_mode": config.sigma_mode,
        "sigma_min": config.sigma_min if config.sigma_mode == "uniform" else None,
        "sigma_max": config.sigma_max if config.sigma_mode == "uniform" else None,
        "fixed_sigma": config.fixed_sigma if config.sigma_mode == "fixed" else None,
        "sigma_law": "Uniform(sigma_min, sigma_max)" if config.sigma_mode == "uniform" else "fixed",
        "sigma_law_is_project_assumption": config.sigma_mode == "uniform",
        "formula": {
            "omega_n": "sqrt(12/n) * (sum(U_k) - n/2), U_k ~ Uniform(0,1)",
            "formula_1": "eta = omega_n - (omega_n^3 - 3*omega_n)/(20*n)",
            "formula_2": "eta = omega_n - 41*(omega_n^5 - 10*omega_n^3 + 15*omega_n)/(13440*n^2)",
            "center": "x0_i = center_mean + center_std * eta_i",
        },
    }
    metadata["generation_rows"] = generation_rows
    return dataset, metadata


def config_to_dict(config: GenerationConfig) -> dict:
    """Сериализуемые параметры конфигурации."""
    return asdict(config)
