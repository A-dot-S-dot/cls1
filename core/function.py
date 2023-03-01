import numpy as np


def maximum(*values, **kwargs) -> np.ndarray:
    if len(values) == 1:
        return values[0]
    else:
        return np.maximum(values[0], maximum(*values[1:]), **kwargs)


def minimum(*values, **kwargs) -> np.ndarray:
    if len(values) == 1:
        return values[0]
    else:
        return np.minimum(values[0], minimum(*values[1:]), **kwargs)


def minmod(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    minmod = np.zeros(a.shape)
    case = np.sign(a) == np.sign(b)

    minmod[case] = np.sign(a[case]) * np.minimum(np.abs(a[case]), np.abs(b[case]))

    return minmod
