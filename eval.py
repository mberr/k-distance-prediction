# coding=utf-8
"""
Evaluation utilities.
"""
from typing import Union

import numpy
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors.ball_tree import BallTree
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor


def get_candidate_set_size(
        x_eval: numpy.ndarray,
        lower_bound: numpy.ndarray,
        upper_bound: numpy.ndarray,
        range_index: BallTree,
) -> numpy.ndarray:
    """
    Computes the candidate set size with index support.

    :param x_eval: numpy.ndarray, shape: (n_eval, d), dtype: numpy.float
        The data points to evaluate.
    :param lower_bound: numpy.ndarray, shape: (n_eval,), dtype: numpy.float
        The lower bounds (one per data point)
    :param upper_bound: numpy.ndarray, shape: (n_eval,), dtype: numpy.float
        The upper bounds (one per data point)
    :param range_index: BallTree
        The index.

    :return: numpy.ndarray, shape: (n_eval,), dtype: numpy.int
        The candidate set sizes per data point.
    """
    if lower_bound.ndim == 2:
        n, k = lower_bound.shape
        candidate_set_size = numpy.empty(shape=(n, k), dtype=numpy.int32)
        for i in range(k):
            candidate_set_size[:, i] = get_candidate_set_size(x_eval=x_eval, lower_bound=lower_bound[..., i], upper_bound=upper_bound[..., i], range_index=range_index)
    elif lower_bound.ndim == 1:

        # Distance smaller than lower bound? -> true inclusion
        n_lower = range_index.query_radius(X=x_eval, r=lower_bound, count_only=True)

        # Distance larger than upper bound? -> true exclusion
        n_upper = range_index.query_radius(X=x_eval, r=upper_bound, count_only=True)

        # Distance between? -> candidate
        candidate_set_size = n_upper - n_lower
    else:
        raise ValueError()

    return candidate_set_size


def get_model_size(model: Union[DecisionTreeRegressor, MultiOutputRegressor, GradientBoostingRegressor, MLPRegressor]) -> int:
    """
    Calculate the number of parameters of a model.

    :param model: DecisionTreeRegressor | MultiOutputRegressor | GradientBoostingRegressor | MLPRegressor
        The model

    :return: int
        The number of parameters needed by the model.
    """
    if isinstance(model, DecisionTreeRegressor):
        tree = model.tree_
        node_count = tree.node_count
        node_count_leaves = (node_count + 1) // 2
        node_count_inner = node_count - node_count_leaves
        _, n_outputs, _ = tree.value.shape

        # inner nodes store: threshold (float), attribute (int), 2*children (int)
        # leaf nodes store: #outputs values (float)

        n_parameters = node_count_leaves * n_outputs + node_count_inner * 4
    elif isinstance(model, MultiOutputRegressor):
        n_parameters = sum(get_model_size(sub_model) for sub_model in model.estimators_)
    elif isinstance(model, GradientBoostingRegressor):
        n_parameters = model.n_estimators + sum(get_model_size(sub_model) for sub_model in model.estimators_[:, 0])
    elif isinstance(model, MLPRegressor):
        n_parameters = sum(c.size for c in model.coefs_) + sum(i.size for i in model.intercepts_)
    else:
        raise ValueError(f'Unknown model type {type(model)}')

    return n_parameters
