from typing import Tuple

from .index_mapping import IndexMapping


class FineGridIndicesMapping(IndexMapping):
    coarsening_degree: int

    def __init__(self, fine_mesh_size: int, coarsening_degree: int):
        self.coarsening_degree = coarsening_degree
        self._assert_admissible_coarsening_degree(fine_mesh_size)

    def _assert_admissible_coarsening_degree(self, mesh_size: int):
        if mesh_size % self.coarsening_degree != 0:
            raise ValueError(
                f"Mesh size of {mesh_size} is not divisible with respect to coarsening degree {self.coarsening_degree}."
            )

    def __call__(self, coarsen_cell_index: int) -> Tuple[int, ...]:
        return tuple(
            [
                self.coarsening_degree * coarsen_cell_index + i
                for i in range(self.coarsening_degree)
            ]
        )
