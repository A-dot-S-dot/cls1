from typing import Tuple

import numpy as np

from .system_flux import SystemFlux


class CorrectedFlux(SystemFlux):
    """Adds to a given flux a subgrid flux."""

    coarse_flux: SystemFlux
    subgrid_flux: SystemFlux

    def __call__(self, dof_vector: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        flux = self.coarse_flux(dof_vector)
        subgrid_flux = self.subgrid_flux(dof_vector)
        return flux[0] + subgrid_flux[0], flux[1] + subgrid_flux[1]
