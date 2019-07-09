# coding=utf-8
"""
Model Selection.
"""
import argparse
import logging
import os
import pickle
from os import path

import h5py
import numpy
import pandas
import tqdm
from scipy.spatial.qhull import ConvexHull
from sklearn import metrics
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor

from eval import get_candidate_set_size, get_model_size
from linear_approx import MRkNNCoPTree
from persistence import load_csr_from_hdf


def process_dataset(
        data_set_name: str,
        index_root: str,
        model_root: str,
        output_root: str,
):
    all_result_path = path.join(output_root, f'{data_set_name}.all.pkl')
    if path.isfile(all_result_path):
        logging.info(f'Skipping already processed dataset {data_set_name}.')
        return

    # Load data
    logging.info('Loading data.')
    with h5py.File(path.join(index_root, f'{data_set_name}.h5'), mode='r') as h5f:
        d = load_csr_from_hdf(h5f=h5f, key='distances')
    kd = d.data.reshape([d.shape[0], -1])
    skd = numpy.sort(kd, axis=-1)

    with open(path.join(index_root, f'{data_set_name}.index'), 'rb') as pf:
        index = pickle.load(pf)
    x = index.get_arrays()[0].astype(numpy.float32)

    with open(path.join(model_root, f'{data_set_name}.mrknn'), 'rb') as pf:
        mrknn: MRkNNCoPTree = pickle.load(pf)

    mrknn_coefficients = mrknn.coefficients
    mrknn_size = mrknn.coefficients.size
    logging.info(f'MRkNNCoP tree has {mrknn_size} parameters.')

    # Define search space
    models = {}

    # Gradient Boosting
    models.update({
        ('gradient_boosting', (('learning_rate', learning_rate), ('max_depth', max_depth), ('n_estimators', n_estimators))): MultiOutputRegressor(estimator=GradientBoostingRegressor(learning_rate=learning_rate, max_depth=max_depth, n_estimators=n_estimators)) for max_depth in (2, 3, 5, 8) for learning_rate in (1.0, 0.1, 0.01) for n_estimators in (5, 20, 100, 1000)
    })
    # Regression Tree
    models.update({
        ('decision_tree_regressor', (('max_depth', max_depth), ('min_samples_leaf', min_samples_leaf))): DecisionTreeRegressor(max_depth=max_depth, min_samples_leaf=min_samples_leaf) for max_depth in (1, 2, 3, 5, 8, 13, 21) for min_samples_leaf in (1, 2, 3, 5, 8, 13, 21)
    })
    # MLP: 1-3 layers
    models.update({
        ('mlp', ('hidden_layer_sizes', hidden_layer_sizes)): MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, max_iter=1024) for a in (16, 32, 64, 128, 256) for hidden_layer_sizes in ((a,), (a, a), (a, a, a))
    })
    # MLP: 4-8 layers
    models.update({
        ('mlp', ('hidden_layer_sizes', hidden_layer_sizes)): MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, max_iter=1024) for hidden_layer_sizes in ((4, 8, 16, 32), (4, 4, 8, 8, 16, 16, 16), (4, 8, 8, 8, 8, 16))
    })
    logging.info(f'The search space comprises {len(models)} different models.')

    # Train models
    logging.info('Training models.')
    real_coef_flat = mrknn_coefficients.reshape((-1, 4))
    data = []
    for spec, model in tqdm.tqdm(models.items(), total=len(models), unit='model', unit_scale=True):
        model.fit(x, real_coef_flat)
        coef_pred_flat = model.predict(x)
        model_size = get_model_size(model)
        mae = metrics.mean_absolute_error(coef_pred_flat, real_coef_flat)
        data.append([model_size, model_size / mrknn_size, mae, spec[0], spec[1]])
    all_model_df = pandas.DataFrame(data, columns=['size', 'rel_size', 'mae', 'type', 'spec'])
    all_model_df.to_csv(path.join(output_root, f'{data_set_name}.all_models.csv'))

    # Prune models based on relative size and MAE
    logging.info('Pruning models.')
    max_rel_size = 1.0
    max_mae = 1.0
    selection = all_model_df[(all_model_df['rel_size'] < max_rel_size) & (all_model_df['mae'] < max_mae)]

    # Get skyline
    values = selection[['mae', 'rel_size']].values
    hull = ConvexHull(values)
    skyline = []
    for i in hull.vertices:
        mae, rel_size = values[i]
        if not ((values[:, 0] < mae) & (values[:, 1] < rel_size)).any():
            skyline.append(i)
    skyline = selection.iloc[skyline]
    skyline.sort_values(by='mae', inplace=True)

    # Compute candidate set size for MRkNNCoP tree
    mrknn_lower, mrknn_upper = mrknn.predict_bounds()
    mrknn_candidate_set_size = get_candidate_set_size(x_eval=x, lower_bound=mrknn_lower, upper_bound=mrknn_upper, range_index=index)

    # Prepare data evaluation data results
    eval_df_data = []
    candidate_set_sizes = {}

    # Prepare MRkNNCoP model
    mrknn_pred = MRkNNCoPTree()
    logging.info('Evaluating candidate set size.')
    for tup in tqdm.tqdm(skyline.itertuples(), total=len(skyline)):
        # Get trained coefficient regressor
        key = (tup.type, tup.spec)
        coef_regressor = models.get(key)

        # Predict coefficients
        coef_pred = coef_regressor.predict(x).reshape((-1, 2, 2))

        # Sanitise slope coefficient
        coef_pred[:, :, 1] = numpy.maximum(coef_pred[:, :, 1], 0.0)

        # Plug-in coefficients into MRkNNCoP tree model
        mrknn_pred.coefficients = coef_pred

        # Predict bounds
        mrknn_pred_lower, mrknn_pred_upper = mrknn_pred.predict_bounds()

        # Compute maximum training error in both directions
        mrknn_pred_lower_error = numpy.maximum(mrknn_pred_lower - skd, 0.0)
        mrknn_pred_upper_error = numpy.maximum(skd - mrknn_pred_upper, 0.0)
        pred_lower_error_max = mrknn_pred_lower_error.max(axis=0)
        pred_upper_error_max = mrknn_pred_upper_error.max(axis=0)

        # Compute safe bounds
        mrknn_pred_safe_lower, mrknn_pred_safe_upper = numpy.maximum(mrknn_pred_lower - pred_lower_error_max, 0.0), mrknn_pred_upper + pred_upper_error_max

        # Compute candidate set size
        mrknn_pred_candidate_set_size = get_candidate_set_size(x_eval=x, lower_bound=mrknn_pred_safe_lower, upper_bound=mrknn_pred_safe_upper, range_index=index)
        candidate_set_sizes[key] = mrknn_pred_candidate_set_size

        # Append to evaluation data frame
        eval_df_data.append([tup.type, tup.spec, tup.mae, tup.rel_size, pred_lower_error_max.max(), pred_upper_error_max.max(), mrknn_pred_candidate_set_size.mean()])

    # Compose data frame
    eval_df = pandas.DataFrame(data=eval_df_data, columns=['type', 'spec', 'mae', 'rel_size', 'max_upper', 'max_lower', 'mean_cand_set_size'])
    eval_df.sort_values(by='mean_cand_set_size', inplace=True)
    eval_df.to_csv(path.join(output_root, f'{data_set_name}.eval.csv'))

    logging.info('Saving results.')
    to_save = {}
    for key, cand_set_size in candidate_set_sizes.items():
        model = models[key]
        to_save[key] = {
            'model': model,
            'candidate_set_sizes': cand_set_size,
        }
    to_save['mrknn'] = {
        'model': mrknn,
        'candidate_set_sizes': mrknn_candidate_set_size,
    }
    with open(all_result_path, 'wb') as pf:
        pickle.dump(to_save, pf, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(asctime)s %(message)s')

    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--index_root', type=str, default='./index', help='The index root.')
    parser.add_argument('--model_root', type=str, default='./models', help='The models root.')
    parser.add_argument('--output_root', type=str, default='./results', help='The output root.')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_root, exist_ok=True)

    # Auto-discover datasets
    datasets = list(map(lambda s: s.replace('.mrknn', ''), filter(lambda s: s.endswith('.mrknn'), os.listdir('./models'))))
    for i, data_set_name in enumerate(datasets):
        logging.info(f'[{i + 1}/{len(datasets)}] Processing dataset: {data_set_name}')
        process_dataset(
            data_set_name=data_set_name,
            index_root=args.index_root,
            model_root=args.model_root,
            output_root=args.output_root,
        )
