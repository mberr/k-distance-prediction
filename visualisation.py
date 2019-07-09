# coding=utf-8
"""
Visualisation utilities.
"""
import numpy
from matplotlib import pyplot as plt

import settings


def plot_with_quantiles(
        *arrays: numpy.ndarray,
        ci: float = 0.9,
        alpha=0.5,
        plot_kwargs=None,
        colors=None,
) -> None:
    if plot_kwargs is None:
        plot_kwargs = {}
    assert 0.0 <= ci <= 1.0
    c = None
    for i, array in enumerate(arrays):
        assert array.ndim == 2
        x = numpy.arange(1, settings.K_MAX + 1)
        q_low, q_high = 0.5 - 0.5 * ci, 0.5 + 0.5 * ci
        low, high = numpy.quantile(array, q=[q_low, q_high], axis=0)
        if colors is not None:
            c = colors[i]
        base_line, = plt.plot(x, array.mean(axis=0), **plot_kwargs, c=c)
        plt.fill_between(x, low, high, facecolor=base_line.get_color(), alpha=alpha)
    plt.xlim(1, settings.K_MAX)
    ylim = list(plt.ylim())
    ylim[0] = 0
    plt.ylim(ylim)
