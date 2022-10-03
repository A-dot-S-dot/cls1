"""This module provides a task for displaying help messages.

"""
from parser.solver_parser import SOLVER_PARSERS

from .command import Command

BENCHMARK_MESSAGE = """Available benchmarks.

Linear Advection (ut+ux=0)
--------------------------
    0           three hills:    I=[0,1], periodic boundaries, T=1
    1           two hills:      I=[0,1], periodic boundaries, T=1 (plot_default)
    2           one hill:       I=[0,1], periodic boundaries, T=1
    3           Cosine:         u(x)=cos(2*pi*(x-0.5)), I=[0,1], periodic boundaries, T=1 (eoc_default)
    4           Gaussian Bell:  u(x)=exp(-100*(x-0.5)^2), I=[0,1], periodic boundaries, T=1

Burgers
-------
    0           Sinus:          u(x)=sin(2*pi*x), I=[0,1], periodic boundaries, T=0.5 (plot_default)
    1           Sinus:          u(x)=sin(x)+0.5, I=[0,1], periodic boundaries, T=0.5 (eoc_default)

"""


class HelpCommand(Command):
    def execute(self):
        page = self._args.page

        if page in SOLVER_PARSERS.keys():
            SOLVER_PARSERS[page].print_help()
        elif page == "benchmark":
            print(BENCHMARK_MESSAGE)
        else:
            raise NotImplementedError(f"No help message for {page} available.")
