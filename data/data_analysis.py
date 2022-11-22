import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

JUMP = 50


def load_data(data_path):
    return pd.read_csv(data_path, header=[0, 1], skipinitialspace=True, index_col=0)


def get_inner_box(
    data,
    h0_left=None,
    h0_right=None,
    q0_left=None,
    q0_right=None,
    h1_left=None,
    h1_right=None,
    q1_left=None,
    q1_right=None,
):
    h0_left = h0_left or -np.inf
    h0_right = h0_right or np.inf
    q0_left = q0_left or -np.inf
    q0_right = q0_right or np.inf
    h1_left = h1_left or -np.inf
    h1_right = h1_right or np.inf
    q1_left = q1_left or -np.inf
    q1_right = q1_right or np.inf

    data = data[
        (data["U0", "h"] > h0_left)
        & (data["U0", "h"] < h0_right)
        & (data["U0", "q"] > q0_left)
        & (data["U0", "q"] < q0_right)
        & (data["U1", "h"] > h1_left)
        & (data["U1", "h"] < h1_right)
        & (data["U1", "q"] > q1_left)
        & (data["U1", "q"] < q1_right)
    ]

    return data


def get_outer_box(
    data,
    h0_left=None,
    h0_right=None,
    q0_left=None,
    q0_right=None,
    h1_left=None,
    h1_right=None,
    q1_left=None,
    q1_right=None,
):
    h0_left = h0_left or np.inf
    h0_right = h0_right or -np.inf
    q0_left = q0_left or np.inf
    q0_right = q0_right or -np.inf
    h1_left = h1_left or np.inf
    h1_right = h1_right or -np.inf
    q1_left = q1_left or np.inf
    q1_right = q1_right or -np.inf

    data = data[
        (data["U0", "h"] < h0_left)
        & (data["U0", "h"] > h0_right)
        & (data["U0", "q"] < q0_left)
        & (data["U0", "q"] > q0_right)
        & (data["U1", "h"] < h1_left)
        & (data["U1", "h"] > h1_right)
        & (data["U1", "q"] < q1_left)
        & (data["U1", "q"] > q1_right)
    ]

    return data


def plot_projection(
    data, label_1_index, label_2_index, target_label_index, axes=None, jumps=None
):
    axes = axes or plt.figure().add_subplot(projection="3d")
    jumps = jumps or JUMP
    target_label_index += 4

    axes.plot_trisurf(
        data.iloc[::jumps, label_1_index],
        data.iloc[::jumps, label_2_index],
        data.iloc[::jumps, target_label_index],
    )
    axes.set_xlabel(data.columns[label_1_index])
    axes.set_ylabel(data.columns[label_2_index])
    axes.set_zlabel(data.columns[target_label_index])

    return axes


def plot_projections(data, target_label_index, jumps=None):
    jumps = jumps or JUMP

    figure = plt.figure()
    figure.suptitle(f"Every {jumps}-th data point")
    for i, (label_1_index, label_2_index) in enumerate(
        itertools.combinations(range(4), 2)
    ):
        plot_projection(
            data,
            label_1_index,
            label_2_index,
            target_label_index,
            axes=figure.add_subplot(231 + i, projection="3d"),
            jumps=jumps,
        )
