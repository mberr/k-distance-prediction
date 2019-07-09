# coding=utf-8
"""
Unit test for MRkNNCoP tree.
"""
import numpy

import settings
from linear_approx import MRkNNCoPTree, compute_bounds_coefficients

# Allowed deviation
EPSILON = 1.0e-04


def test_compute_bounds():
    numpy.random.seed(42)
    skd = _get_random_data()

    log_d = numpy.log(skd)
    log_k = numpy.log(numpy.arange(1, settings.K_MAX + 1, dtype=numpy.float32))

    coefficients = compute_bounds_coefficients(log_k=log_k, log_d=log_d)

    # Assert slope is non-decreasing
    assert numpy.all(numpy.greater_equal(coefficients[:, :, 1], 0))

    # Check bounds
    bounds = coefficients[:, :, 0, None] + log_k[None, None, :] * coefficients[:, :, 1, None]
    assert numpy.all(numpy.diff(bounds, axis=-1) >= 0)
    lower_bound = bounds[:, 0, :]
    assert numpy.all(lower_bound <= log_d + EPSILON)
    upper_bound = bounds[:, 1, :]
    assert numpy.all(upper_bound >= log_d - EPSILON)
    assert numpy.all(numpy.less_equal(lower_bound, upper_bound))


def test_mrknncop_tree_bounds():
    skd = _get_random_data()

    tree = MRkNNCoPTree()
    tree.fit(y=skd)

    lower_bound, upper_bound = tree.predict_bounds()

    bounds = numpy.stack([lower_bound, upper_bound], axis=1)

    # Check bounds
    assert numpy.all(numpy.diff(bounds, axis=-1) >= 0)
    lower_bound = bounds[:, 0, :]
    assert numpy.all(lower_bound <= skd + EPSILON)
    upper_bound = bounds[:, 1, :]
    assert numpy.all(upper_bound >= skd - EPSILON)
    assert numpy.all(lower_bound <= upper_bound)


def _get_random_data():
    n = 2 * settings.K_MAX
    d = 2
    # Random data
    x = numpy.random.uniform(size=(n, d)).astype(numpy.float32)

    # Compute distances
    d = numpy.linalg.norm(x[:, None, :] - x[None, :, :], axis=-1, keepdims=False)
    assert d.shape == (n, n)

    skd = numpy.sort(d, axis=1)[:, 1:settings.K_MAX + 1]
    return skd
