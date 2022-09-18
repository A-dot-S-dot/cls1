"""This module provides a task for displaying help messages.

"""
from .task import Task
from parser.solver_parser import SOLVER_PARSER

BENCHMARK_MESSAGE = """Available benchmarks.

Linear Advection (ut+ux=0)
--------------------------
    1           Several Funcs:  I=[0,1], periodic boundaries, T=1 (plot_default)
    2           Cosine:         u(x)=cos(2*pi*(x-0.5)), I=[0,1], periodic boundaries, T=1 (eoc_default)
    3           Gaussian Bell:  u(x)=exp(-100*(x-0.5)^2), I=[0,1], periodic boundaries, T=1

Burgers
-------
    1           Sinus:          u(x)=sin(2*pi*x), I=[0,1], periodic boundaries, T=0.5 (plot_default)
    2           Sinus:          u(x)=sin(x)+0.5, I=[0,1], periodic boundaries, T=0.5 (eoc_default)

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
