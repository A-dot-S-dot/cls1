from typing import Callable
import numpy as np


ScalarFunction = Callable[[float], float]
SystemFunction = Callable[[float], np.ndarray]
