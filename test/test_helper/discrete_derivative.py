from math_type import ScalarFunction


def discrete_derivative(f: ScalarFunction, x: float, eps: float = 1e-7):
    return (f(x + eps) - f(x - eps)) / (2 * eps)
