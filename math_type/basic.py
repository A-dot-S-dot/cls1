"""This module provides different types for typing."""
from typing import Callable

from numpy import ndarray

RealD = ndarray

FunctionRealToReal = Callable[[float], float]
FunctionRealToRealD = Callable[[float], RealD]
FunctionRealDToReal = Callable[[RealD], float]
FunctionRealDToRealD = Callable[[RealD], RealD]
