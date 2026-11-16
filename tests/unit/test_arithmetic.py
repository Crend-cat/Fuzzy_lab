import pytest

from fuzzy_core.arithmetic import add, subtract, multiply, divide, compare
from fuzzy_core.level_sets import LevelInterval


def test_interval_addition_and_subtraction():
    a = LevelInterval(1, 3)
    b = LevelInterval(2, 5)
    assert add(a, b, 0.7).interval == LevelInterval(3, 8)
    assert subtract(a, b, 0.7).interval == LevelInterval(-4, 1)


def test_interval_multiplication():
    assert multiply(LevelInterval(-2, 3), LevelInterval(4, 5), 0.7).interval == LevelInterval(-10, 15)


def test_interval_division_and_zero_check():
    assert divide(LevelInterval(2, 4), LevelInterval(1, 2), 0.7).interval == LevelInterval(1, 4)
    with pytest.raises(ZeroDivisionError):
        divide(LevelInterval(2, 4), LevelInterval(-1, 2), 0.7)


def test_interval_comparison():
    assert "левее" in compare(LevelInterval(1, 2), LevelInterval(3, 4))
