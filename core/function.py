import numpy as np


def maximum(*values, **kwargs) -> np.ndarray:
    if len(values) == 1:
        return values[0]
    else:
        return np.maximum(values[0], maximum(*values[1:]))


def minimum(*values, **kwargs) -> np.ndarray:
    if len(values) == 1:
        return values[0]
    else:
        return np.minimum(values[0], minimum(*values[1:]))
