import argparse
from typing import Any

import core
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from finite_volume.shallow_water.solver.reduced_model import Curvature

from .command import Command, CommandParser


def compare_2d_scatter_plots(
    x_llf,
    y_llf,
    x_mcl,
    y_mcl,
    x_es1,
    y_es1,
    xlabel="",
    ylabel="",
    suptitle="",
    save=None,
):
    fig, axs = plt.subplots(3, 1)
    fig.suptitle(suptitle)
    xlim = (
        np.min((np.min(x_llf), np.min(x_mcl), np.min(x_es1))),
        np.max((np.max(x_llf), np.max(x_mcl), np.max(x_es1))),
    )
    ylim = (
        np.min((np.min(y_llf), np.min(y_mcl), np.min(y_es1))),
        np.max((np.max(y_llf), np.max(y_mcl), np.max(y_es1))),
    )

    axs[0].set_title("LLF")
    axs[0].scatter(x_llf, y_llf)
    axs[0].set_xlabel(xlabel)
    axs[0].set_ylabel(ylabel)
    axs[0].set_xlim(xlim[0], xlim[1])
    axs[0].set_ylim(ylim[0], ylim[1])

    axs[1].set_title("MCL")
    axs[1].scatter(x_mcl, y_mcl)
    axs[1].set_xlabel(xlabel)
    axs[1].set_ylabel(ylabel)
    axs[1].set_xlim(xlim[0], xlim[1])
    axs[1].set_ylim(ylim[0], ylim[1])

    axs[2].set_title("ES1")
    axs[2].scatter(x_es1, y_es1)
    axs[2].set_xlabel(xlabel)
    axs[2].set_ylabel(ylabel)
    axs[2].set_xlim(xlim[0], xlim[1])
    axs[2].set_ylim(ylim[0], ylim[1])

    if save:
        fig.savefig(save)


def create_2d_scatter_plots(df_llf, df_mcl, df_es1, save=False):
    compare_2d_scatter_plots(
        df_llf[("k", "h")],
        df_llf[("G_1.5", "h")],
        df_mcl[("k", "h")],
        df_mcl[("G_1.5", "h")],
        df_es1[("k", "h")],
        df_es1[("G_1.5", "h")],
        xlabel="$\kappa_h$",
        ylabel="$G_{1.5}^h$",
        suptitle="Height curvature and subgrid flux",
        save="data/curvature-analysis/kh_Gh.png" if save else None,
    )
    compare_2d_scatter_plots(
        df_llf[("k", "q")],
        df_llf[("G_1.5", "h")],
        df_mcl[("k", "q")],
        df_mcl[("G_1.5", "h")],
        df_es1[("k", "q")],
        df_es1[("G_1.5", "h")],
        xlabel="$\kappa_q$",
        ylabel="$G_{1.5}^h$",
        suptitle="Discharge curvature and height subgrid flux",
        save="data/curvature-analysis/kq_Gh.png" if save else None,
    )
    compare_2d_scatter_plots(
        df_llf[("k", "h")],
        df_llf[("G_1.5", "q")],
        df_mcl[("k", "h")],
        df_mcl[("G_1.5", "q")],
        df_es1[("k", "h")],
        df_es1[("G_1.5", "q")],
        xlabel="$\kappa_h$",
        ylabel="$G_{1.5}^q$",
        suptitle="Height curvature and discharge subgrid flux",
        save="data/curvature-analysis/kh_Gq.png" if save else None,
    )
    compare_2d_scatter_plots(
        df_llf[("k", "q")],
        df_llf[("G_1.5", "q")],
        df_mcl[("k", "q")],
        df_mcl[("G_1.5", "q")],
        df_es1[("k", "q")],
        df_es1[("G_1.5", "q")],
        xlabel="$\kappa_q$",
        ylabel="$G_{1.5}^q$",
        suptitle="Discharge curvature and subgrid flux",
        save="data/curvature-analysis/kq_Gq.png" if save else None,
    )


class PlotCurvatureAgainstSubgridFlux(Command):
    _show: bool
    _save: bool

    def __init__(self, show=True, save=True):
        self._show = show
        self._save = save

    def execute(self):
        df_llf = core.load_data("data/reduced-llf/data.csv")
        df_mcl = core.load_data("data/reduced-mcl/data.csv")
        df_es1 = core.load_data("data/reduced-es1/data.csv")

        curvature = Curvature()
        curvature_llf = curvature.transform(df_llf.values[:, :8])
        curvature.step_length = 1.0
        curvature_mcl = curvature.transform(df_mcl.values[:, :8])
        curvature_es1 = curvature.transform(df_es1.values[:, :8])

        df_llf[("k", "h")] = curvature_llf[:, 8]
        df_llf[("k", "q")] = curvature_llf[:, 9]
        df_llf = df_llf.reindex(
            columns=pd.MultiIndex.from_product(
                [["U0", "U1", "U2", "U3", "k", "G_1.5"], ["h", "q"]]
            )
        )

        df_mcl[("k", "h")] = curvature_mcl[:, 8]
        df_mcl[("k", "q")] = curvature_mcl[:, 9]
        df_mcl = df_mcl.reindex(
            columns=pd.MultiIndex.from_product(
                [["U0", "U1", "U2", "U3", "k", "G_1.5"], ["h", "q"]]
            )
        )

        df_es1[("k", "h")] = curvature_es1[:, 8]
        df_es1[("k", "q")] = curvature_es1[:, 9]
        df_es1 = df_es1.reindex(
            columns=pd.MultiIndex.from_product(
                [["U0", "U1", "U2", "U3", "k", "G_1.5"], ["h", "q"]]
            )
        )

        create_2d_scatter_plots(df_llf, df_mcl, df_es1, self._save)

        if self._show:
            plt.show()
        else:
            plt.close()


class AnalyzeCurvatureParser(CommandParser):
    def _get_parser(self, parsers) -> Any:
        return parsers.add_parser(
            "analyze-curvature",
            help="Analyze curvature.",
            description="""Analyze curvature by plotting it against subgrid flux.""",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )

    def _add_arguments(self, parser):
        parser.add_argument(
            "--hide",
            help=f"Do not show any figures.",
            action="store_true",
        )
        parser.add_argument("--save", help="Save plots.", action="store_true")

    def postprocess(self, arguments):
        arguments.show = not arguments.hide
        arguments.command = PlotCurvatureAgainstSubgridFlux

        del arguments.hide
