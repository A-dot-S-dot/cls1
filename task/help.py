"""This module provides a task for displaying help messages.

"""
from .task import Task
from parser.solver_parser import SOLVER_PARSER

BENCHMARK_MESSAGE = """Available benchmarks.

Linear Advection (ut+ux=0)
--------------------------
    rect        Rectangle:      I=[0,1], periodic boundaries, T=1 (plot_default)
    cos         Cosine:         u(x)=cos(2*pi*(x-0.5)), I=[0,1], periodic boundaries, T=1 (eoc_default)
    gauss       Gaussian Bell:  u(x)=exp(-100*(x-0.5)^2), I=[0,1], periodic boundaries, T=1

Burgers
-------
    sin         Sinus:          u(x)=sin(2*pi*x), I=[0,1], periodic boundaries, T=0.1 (plot_default)

Note, no 'eoc_default' is implemented.


"""


class HelpTask(Task):
    def execute(self):
        page = self._args.page

        if page in SOLVER_PARSER.keys():
            SOLVER_PARSER[page].print_help()
        elif page == "benchmark":
            print(BENCHMARK_MESSAGE)
        else:
            raise NotImplementedError(f"No help message for {page} available.")
