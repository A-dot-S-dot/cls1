import argparse
from typing import Any

import core
from finite_volume.shallow_water.solver.reduced_model import Curvature

from .command import Command, CommandParser


class DeleteCurvature(Command):
    _data_path: str
    _step_length: float
    _target_path: str
    _curvature_height_threshold: float
    _curvature_discharge_threshold: float

    def __init__(
        self,
        data_path: str,
        target_path: str,
        step_length=None,
        curvature_height_threshold=None,
        curvature_discharge_threshold=None,
    ):
        self._data_path = data_path
        self._target_path = target_path
        self._step_length = step_length or 2.0
        self._curvature_height_threshold = curvature_height_threshold or 0.25
        self._curvature_discharge_threshold = curvature_discharge_threshold or 0.5

    def execute(self):
        df = core.load_data(self._data_path)
        curvature = Curvature(step_length=self._step_length).transform(
            df.values[:, :8]
        )[:, 8:]

        df = df[
            (curvature[:, 0] < self._curvature_height_threshold)
            & (curvature[:, 1] < self._curvature_discharge_threshold)
        ]

        core.save_data(df, self._target_path)


class DeleteCurvatureParser(CommandParser):
    def _get_parser(self, parsers) -> Any:
        return parsers.add_parser(
            "delete-curvature",
            help="Delete data which exceed certain curvature.",
            description="""Delete data which exceed certain curvature.""",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )

    def _add_arguments(self, parser):
        self._add_data_path(parser)
        self._add_target_path(parser)
        self._add_step_length(parser)
        self._add_curvature_height_threshold(parser)
        self._add_curvature_discharge_threshold(parser)

    def _add_data_path(self, parser):
        parser.add_argument("data_path", help="Which data should be modified.")

    def _add_target_path(self, parser):
        parser.add_argument(
            "target_path", help="Where the modified data should be stored."
        )

    def _add_step_length(self, parser):
        parser.add_argument(
            "-s",
            "--step-length",
            help="Step size of the coarse grid associated with the data.",
            type=core.positive_float,
            metavar="<h>",
        )

    def _add_curvature_height_threshold(self, parser):
        parser.add_argument(
            "--kh",
            help="Specify height curvature threshold.",
            type=core.positive_float,
            dest="curvature_height_threshold",
            metavar="<kh>",
        )

    def _add_curvature_discharge_threshold(self, parser):
        parser.add_argument(
            "--kq",
            help="Specify discharge curvature threshold.",
            type=core.positive_float,
            dest="curvature_discharge_threshold",
            metavar="<kq>",
        )

    def postprocess(self, arguments):
        arguments.command = DeleteCurvature
