import argparse
from typing import Any, Iterable, List, Optional

import core
import defaults
import numpy as np
from benchmark import shallow_water
from benchmark.shallow_water import HEIGHT_AVERAGE
from tqdm.auto import tqdm, trange

from .calculate import CalculateParser
from .command import Command
from .error_evolution import (
    GenerateShallowWaterErrorEvolutionSeries,
    PlotShallowWaterAverageErrorEvolution,
)


class ParameterVariation:
    values: Iterable
    name: str
    short: str
    description: str
    benchmark_defaults = {}


class HeightAverageVariation(ParameterVariation):
    values = np.array([0.8, 0.9, 1.0, 1.1, 1.2]) * HEIGHT_AVERAGE
    name = "height_average"
    short = "H0"
    description = "Calculate Errors (H0-variation)"
    benchmark_defaults = {"height_wave_number": 2}


class WaveNumberVariation(ParameterVariation):
    values = [1, 2, 5, 6, 9, 10, 15]
    name = "height_wave_number"
    short = "kh"
    description = "Calculate Errors (kh-variation)"


class AmplitudeVariation(ParameterVariation):
    values = [0.6, 0.8, 1.0, 1.5]
    name = "height_amplitude"
    short = "A"
    description = "Calculate errors (A-variation)"


PARAMETER_VARIATIONS = {
    "height-average": HeightAverageVariation(),
    "wave-number": WaveNumberVariation(),
    "amplitude": AmplitudeVariation(),
}


class ParameterVariationTest(Command):
    _directory: str
    _initial_data_number: int
    _seed: Optional[int]
    _end_time: Optional[float]
    _build_references: bool
    _parameter_variations: List[ParameterVariation]
    _save_plot: bool
    _save_animation: bool

    def __init__(
        self,
        solver_approximation: core.Solver,
        solver_exact: core.Solver,
        directory: str,
        initial_data_number=None,
        seed=None,
        end_time=None,
        build_references=False,
        parameter_variations=None,
        save_plot=False,
        save_animation=False,
    ):
        self._solver_approximation = solver_approximation
        self._solver_exact = solver_exact
        self._directory = directory
        self._initial_data_number = initial_data_number or defaults.INITIAL_DATA_NUMBER
        self._seed = seed
        self._end_time = end_time
        self._build_references = build_references
        self._parameter_variations = parameter_variations or []
        self._save_plot = save_plot
        self._save_animation = save_animation

    def execute(self):
        if self._build_references:
            self._build_reference_errors()

        for parameter_variation in tqdm(
            self._parameter_variations,
            desc="Parameter Variation Test",
            unit="test",
            disable=len(self._parameter_variations) == 0,
        ):
            self._execute_parameter_variation_test(parameter_variation)

    def _build_reference_errors(self):
        for _ in trange(1, desc="Build reference errors"):
            generator = GenerateShallowWaterErrorEvolutionSeries(
                self._solver_approximation,
                self._solver_exact,
                seed=self._seed,
                end_time=self._end_time,
                save_directory=self._directory + "reference",
                save_animation=self._save_animation,
                save_plot=self._save_plot,
                save_error=True,
                initial_conditions=self._initial_data_number,
                description="reference",
            )
            generator.execute()

            PlotShallowWaterAverageErrorEvolution(
                generator.times,
                generator.errors,
                show=False,
                save=self._directory + f"reference/error_average.png",
            ).execute()

    def _execute_parameter_variation_test(
        self, parameter_variation: ParameterVariation
    ):
        name = parameter_variation.name
        short = parameter_variation.short

        for param in tqdm(
            parameter_variation.values,
            desc=parameter_variation.description,
            unit=name,
        ):
            generator = GenerateShallowWaterErrorEvolutionSeries(
                self._solver_approximation,
                self._solver_exact,
                seed=self._seed,
                end_time=self._end_time,
                save_directory=self._directory + f"{name}/{param}",
                save_animation=self._save_animation,
                save_plot=self._save_plot,
                save_error=True,
                initial_conditions=self._initial_data_number,
                description=f"${short}={param}$",
                **{name: param},
                **parameter_variation.benchmark_defaults,
            )
            generator.execute()

            PlotShallowWaterAverageErrorEvolution(
                generator.times,
                generator.errors,
                show=False,
                save=self._directory + f"{name}/{short}_{param}_average.png",
            ).execute()


class ParameterVariationTestParser(CalculateParser):
    def _get_parser(self, parsers) -> Any:
        return parsers.add_parser(
            "parameter-variation-test",
            help="Variate parameters and obtain errors.",
            description="""Variate parameters for certain benchmarks, calculate
            its errors and optionally plot, animate the solutions. Note the
            second solver is the reference solver, which ideally calculates the
            best solution.""",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )

    def _add_arguments(self, parser):
        self._add_shallow_water_solver(parser)
        self._add_end_time(parser)
        self._add_directory(parser)
        self._add_build_references(parser)
        self._add_parameter_variations(parser)
        self._add_initial_data_number(parser)
        self._add_seed(parser)
        self._add_save_plot(parser)
        self._add_save_animation(parser)
        self._add_general_arguments(parser)

    def _add_directory(self, parser):
        parser.add_argument("directory", help="Specify where to save test results.")

    def _add_build_references(self, parser):
        parser.add_argument(
            "--build-references",
            help="Build a reference set of errors.",
            action="store_true",
        )

    def _add_parameter_variations(self, parser):
        parser.add_argument(
            "-v",
            "--variations",
            help="Specify which variations should be performed. Available variations are: "
            + ", ".join(["all", *PARAMETER_VARIATIONS.keys()]),
            nargs="*",
            metavar="<variation>",
        )

    def _add_initial_data_number(self, parser):
        parser.add_argument(
            "-n",
            "--initial-data-number",
            help="Specify how many problems should be considered for each parameter value.",
            type=core.positive_int,
            default=defaults.INITIAL_DATA_NUMBER,
            metavar="<number>",
        )

    def _add_seed(self, parser):
        parser.add_argument(
            "--seed",
            help="Seed for generating random benchmarks",
            type=core.positive_int,
            default=defaults.SEED,
            metavar="<seed>",
        )

    def _add_save_plot(self, parser):
        parser.add_argument(
            "--save-plot",
            help="Save plots of both solutions at end time.",
            action="store_true",
        )

    def _add_save_animation(self, parser):
        parser.add_argument(
            "--save-animation",
            help="Save animations of both solutions.",
            action="store_true",
        )

    def postprocess(self, arguments):
        self._assert_two_solver(arguments)
        self._add_save_history_argument(arguments)
        arguments.benchmark = shallow_water.OscillationNoTopographyBenchmark()
        self._adjust_end_time(arguments)
        self._build_solver(arguments)
        self._build_variations(arguments)

        arguments.command = ParameterVariationTest

        del arguments.benchmark

    def _assert_two_solver(self, arguments):
        solver_num = 0 if arguments.solver is None else len(arguments.solver)
        assert (
            solver_num == 2
        ), f"Exactly two solver must be given. There are {solver_num}."

    def _build_solver(self, arguments):
        super()._build_solver(arguments)
        arguments.solver_approximation = arguments.solver[0]
        arguments.solver_exact = arguments.solver[1]

        del arguments.solver

    def _build_variations(self, arguments):
        if arguments.variations is None:
            arguments.parameter_variations = []
        elif "all" in arguments.variations:
            arguments.parameter_variations = PARAMETER_VARIATIONS.values()
        else:
            arguments.parameter_variations = [
                PARAMETER_VARIATIONS[key] for key in arguments.variations
            ]

        del arguments.variations
