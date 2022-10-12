from typing import Callable
import numpy as np

ScalarFunction = Callable[[float], float]
SystemFunction = Callable[[float], np.ndarray]


def positive_float(string: str) -> float:
    value = float(string)
    if value <= 0:
        raise ValueError(f"{value} is not a positive floating point number")

    return value


def positive_int(string: str) -> int:
    value = int(string)
    if value <= 0:
        raise ValueError(f"{value} is not a positive integer")

    return value


def percent_number(string: str) -> float:
    value = float(string)
    if value < 0 or value > 1:
        raise ValueError(f"{value} is not an element of the interval [0,1]")

    return value


def gradient_approximation_type(approximation_type: str) -> str:
    if approximation_type not in ["nodal", "l2_projection"]:
        raise ValueError(f"{approximation_type} must be 'nodal' or 'l2_projection'.")

    return approximation_type
