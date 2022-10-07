from abc import ABC, abstractmethod
from typing import Tuple


class IndexMapping(ABC):
    """Returns indices for a given set of indices."""

    @abstractmethod
    def __call__(self, *index: int) -> Tuple[int, ...]:
        ...
