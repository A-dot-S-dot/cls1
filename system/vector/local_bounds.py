from .dof_vector import DOFVector
from .system_vector import SystemVector


class LocalMaximum(SystemVector):
    """Local maximum vector. Consider a DOF vector (ui). Then, the local maximum
    vector is defined via

        ui_max = max(uj, j is neighbour of i).

    Not assembled by default.

    """

    _dof_vector: DOFVector

    def __init__(self, dof_vector: DOFVector):
        SystemVector.__init__(self, dof_vector.element_space)
        self._dof_vector = dof_vector
        dof_vector.register_observer(self)

    def assemble(self):
        for i in range(self.dimension):
            self[i] = max(
                {
                    self._dof_vector[j]
                    for j in self.element_space.get_neighbour_indices(i)
                }
            )


class LocalMinimum(LocalMaximum):
    """Local maximum vector. Consider a DOF vector (ui). Then, the local maximum
    vector is defined via

        ui_max = max(uj, j is neighbour of i).

    Not assembled by default.

    """

    def __init__(self, dof_vector: DOFVector):
        LocalMaximum.__init__(self, dof_vector)

    def assemble(self):
        for i in range(self.dimension):
            self[i] = min(
                {
                    self._dof_vector[j]
                    for j in self.element_space.get_neighbour_indices(i)
                }
            )
