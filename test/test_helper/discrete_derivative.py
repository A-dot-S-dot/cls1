from math_type import FunctionRealToReal


def discrete_derivative(f: FunctionRealToReal, x: float, eps: float = 1e-7):
    return (f(x + eps) - f(x - eps)) / (2 * eps)
