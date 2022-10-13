import os
import time
from argparse import Namespace
from typing import List, Tuple

import numpy as np
import pandas as pd
from pde_solver.discrete_solution import CoarseSolution, DiscreteSolution
from pde_solver.solver.finite_volume import SWEGodunovSolver
from pde_solver.solver_components import SolverComponents
from pde_solver.solver_space.finite_volume import CoarsenedFiniteVolumeSpace
from pde_solver.system_vector import SWEGodunovNumericalFlux
from tqdm import tqdm

from .command import Command


class SaveCoarseSolutionAndSubgridFluxes(Command):
    """Save coarsened solution and subgrid fluxes in certain file.

    Only available for Shallow-Water. Algorithm assumes FL_{i+1/2}=FR_{i+1/2}.

    """

    _components: SolverComponents
    _coarsening_degree: int
    _local_degree: int
    _fine_numerical_fluxes: List
    _coarse_numerical_flux: SWEGodunovNumericalFlux

    def __init__(self, args: Namespace):
        self._args = args
        self._components = SolverComponents(args)
        self._coarsening_degree = self._args.save.coarsening_degree
        self._local_degree = self._args.save.local_degree

    def execute(self):
        for solver_factory in tqdm(
            self._components.solver_factories,
            desc="Calculate solutions",
            unit="solver",
            leave=False,
        ):
            solver = solver_factory.solver

            start_time = time.time()
            solver.solve()

            tqdm.write(
                f"Solved {solver_factory.info} with {solver_factory.dimension} DOFs and {solver.time_stepping.time_steps} time steps in {time.time()-start_time:.2f}s."
            )

            if isinstance(solver, SWEGodunovSolver):
                self._build_numerical_fluxes(solver)
            else:
                raise NotImplementedError

            coarse_solution = CoarseSolution(solver.solution, self._coarsening_degree)
            left_subgrid_flux, right_subgrid_flux = self._get_subgrid_fluxes(
                coarse_solution
            )

            self._save(coarse_solution, left_subgrid_flux, right_subgrid_flux)

    def _build_numerical_fluxes(self, solver: SWEGodunovSolver):
        self._fine_numerical_fluxes = solver.numerical_fluxes
        self._coarse_numerical_flux = SWEGodunovNumericalFlux()
        self._coarse_numerical_flux.volume_space = CoarsenedFiniteVolumeSpace(
            solver.numerical_flux.volume_space, self._coarsening_degree
        )
        self._coarse_numerical_flux.gravitational_acceleration = (
            solver.numerical_flux.gravitational_acceleration
        )

        # TODO should depend on every benchmark's topography
        self._coarse_numerical_flux.bottom_topography = np.zeros(
            self._coarse_numerical_flux.volume_space.dimension
        )

    def _get_subgrid_fluxes(
        self, coarse_solution: CoarseSolution
    ) -> Tuple[np.ndarray, np.ndarray]:
        left_subgrid_flux = np.array(
            [
                self._get_left_subgrid_flux(time_index, coarse_solution)
                for time_index in range(len(coarse_solution.time) - 1)
            ]
        )

        right_subgrid_flux = np.roll(left_subgrid_flux, -1, axis=1)

        return (left_subgrid_flux, right_subgrid_flux)

    def _get_left_subgrid_flux(
        self,
        time_index: int,
        coarse_solution: CoarseSolution,
    ) -> np.ndarray:
        left_coarse_flux = self._coarse_numerical_flux(
            coarse_solution.values[time_index]
        )[0]
        left_fine_flux = self._fine_numerical_fluxes[time_index][0]

        return left_fine_flux[:: self._coarsening_degree] - left_coarse_flux

    def _save(
        self,
        coarse_solution: DiscreteSolution,
        left_subgrid_flux: np.ndarray,
        right_subgrid_flux: np.ndarray,
    ):
        data_frame = pd.DataFrame(columns=self._create_columns())
        for time_index in range(len(coarse_solution.time) - 1):
            data_frame = self._append_values(
                data_frame,
                coarse_solution.values[time_index],
                left_subgrid_flux[time_index],
                right_subgrid_flux[time_index],
            )

        self._save_data_frame(data_frame)

    def _create_columns(self) -> pd.MultiIndex:
        return pd.MultiIndex.from_product(
            [[*self._get_value_labels(), "GLi", "GRi"], ["h", "q"]],
        )

    def _get_value_labels(self) -> List[str]:
        center_label = "Ui"
        left_labels = ["Ui+" + str(i + 1) for i in range(self._local_degree)]
        right_labels = [
            "Ui-" + str(self._local_degree - i) for i in range(self._local_degree)
        ]

        return [*right_labels, center_label, *left_labels]

    def _append_values(
        self,
        data_frame: pd.DataFrame,
        coarse_solution: np.ndarray,
        left_subgrid_flux: np.ndarray,
        right_subgrid_flux: np.ndarray,
    ) -> pd.DataFrame:
        new_data_frame = pd.DataFrame(columns=self._create_columns())
        new_data_frame["GLi", "h"] = left_subgrid_flux[:, 0]
        new_data_frame["GLi", "q"] = left_subgrid_flux[:, 1]
        new_data_frame["GRi", "h"] = right_subgrid_flux[:, 0]
        new_data_frame["GRi", "q"] = right_subgrid_flux[:, 1]
        new_data_frame["Ui", "h"] = coarse_solution[:, 0]
        new_data_frame["Ui", "q"] = coarse_solution[:, 1]

        for i in range(self._local_degree):
            new_data_frame[f"Ui+" + str(i + 1), "h"] = np.roll(
                coarse_solution[:, 0], shift=-1
            )
            new_data_frame[f"Ui+" + str(i + 1), "q"] = np.roll(
                coarse_solution[:, 1], shift=-1
            )
            new_data_frame[f"Ui-" + str(i + 1), "h"] = np.roll(
                coarse_solution[:, 0], shift=1
            )
            new_data_frame[f"Ui-" + str(i + 1), "q"] = np.roll(
                coarse_solution[:, 1], shift=1
            )

        return pd.concat([data_frame, new_data_frame])

    def _save_data_frame(self, data_frame: pd.DataFrame):
        output_path = "/home/alexey/Projects/network-for-subgrid-fluxes/data.csv"
        data_frame.to_csv(output_path, mode="a", header=not os.path.exists(output_path))
