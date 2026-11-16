"""Интервальная арифметика для множеств фиксированного уровня h."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .level_sets import LevelInterval

Operation = Literal["add", "subtract", "multiply", "divide"]


@dataclass(frozen=True, slots=True)
class IntervalOperationResult:
    operation: Operation
    level: float
    interval: LevelInterval
    explanation: str


def add(first: LevelInterval, second: LevelInterval, h: float) -> IntervalOperationResult:
    result = LevelInterval(first.left + second.left, first.right + second.right)
    return IntervalOperationResult("add", h, result, "Сложение интервалов уровня h")


def subtract(first: LevelInterval, second: LevelInterval, h: float) -> IntervalOperationResult:
    result = LevelInterval(first.left - second.right, first.right - second.left)
    return IntervalOperationResult("subtract", h, result, "Вычитание интервалов уровня h")


def multiply(first: LevelInterval, second: LevelInterval, h: float) -> IntervalOperationResult:
    products = (
        first.left * second.left,
        first.left * second.right,
        first.right * second.left,
        first.right * second.right,
    )
    result = LevelInterval(min(products), max(products))
    return IntervalOperationResult("multiply", h, result, "Умножение интервалов уровня h")


def divide(first: LevelInterval, second: LevelInterval, h: float) -> IntervalOperationResult:
    if second.left <= 0.0 <= second.right:
        raise ZeroDivisionError("Интервал знаменателя содержит ноль; деление не определено")
    quotients = (
        first.left / second.left,
        first.left / second.right,
        first.right / second.left,
        first.right / second.right,
    )
    result = LevelInterval(min(quotients), max(quotients))
    return IntervalOperationResult("divide", h, result, "Деление интервалов уровня h")


def compare(first: LevelInterval, second: LevelInterval) -> str:
    """Определить взаимное положение двух интервалов."""
    if first.right < second.left:
        return "Первый интервал полностью левее второго"
    if first.left > second.right:
        return "Первый интервал полностью правее второго"
    if first.left == second.left and first.right == second.right:
        return "Интервалы совпадают"
    return "Интервалы пересекаются; строгий порядок не установлен"


def apply_operation(
    operation: Operation,
    first: LevelInterval,
    second: LevelInterval,
    h: float,
) -> IntervalOperationResult:
    functions = {"add": add, "subtract": subtract, "multiply": multiply, "divide": divide}
    try:
        function = functions[operation]
    except KeyError as exc:
        raise ValueError(f"Неизвестная операция: {operation}") from exc
    return function(first, second, h)
