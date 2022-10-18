from abc import ABC, abstractmethod


class SourceTermDiscretization(ABC):
    step_length: float

    @abstractmethod
    def __call__(
        self, left_height: float, right_height: float, topography_step: float
    ) -> float:
        ...


class NaturalSourceTermDiscretization(SourceTermDiscretization):
    def __call__(
        self, left_height: float, right_height: float, topography_step: float
    ) -> float:
        return (left_height + right_height) / (2 * self.step_length) * topography_step


class WetDryPreservingSourceTermDiscretization(SourceTermDiscretization):
    def __call__(
        self, left_height: float, right_height: float, topography_step: float
    ) -> float:

        if topography_step >= 0:
            return (
                (left_height + right_height)
                / (2 * self.step_length)
                * min(left_height, topography_step)
            )
        else:
            return (
                (left_height + right_height)
                / (2 * self.step_length)
                * max(-right_height, topography_step)
            )
