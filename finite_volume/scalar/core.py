from typing import Callable, Tuple

import numpy as np

import core


def get_flux(problem: str) -> core.FLUX:
    if problem == "advection":
        return lambda u: u
    elif problem == "burgers":
        return lambda u: 1 / 2 * u**2
    else:
        raise ValueError(f"No flux for '{problem}' defined.")


class WaveSpeed:
    _flux_prime: Callable

    def __init__(self, flux_prime: Callable):
        self._flux_prime = flux_prime

    def __call__(self, value_left, value_right) -> Tuple:
        wave_speed = np.maximum(
            self._flux_prime(value_left), self._flux_prime(value_right)
        )

        return -wave_speed, wave_speed


def get_wave_speed(problem: str) -> WaveSpeed:
    if problem == "advection":
        flux_prime = lambda u: np.ones(u.shape)
    elif problem == "burgers":
        flux_prime = lambda u: u
    else:
        raise ValueError(f"No flux for '{problem}' defined.")

    return WaveSpeed(flux_prime)


def get_riemann_solver(problem: str) -> core.RiemannSolver:
    return core.RiemannSolver(get_flux(problem), get_wave_speed(problem))
