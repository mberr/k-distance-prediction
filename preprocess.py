# coding=utf-8
"""
Preprocess data sets by computing BallTree range index and kNN graph, and also fit MRkNNCoP tree.
"""
import argparse
import logging
import os
import pickle
from os import path

import h5py
import numpy
import pandas
from sklearn.datasets import make_blobs
from sklearn.neighbors import BallTree, kneighbors_graph

from linear_approx import MRkNNCoPTree
from persistence import load_csr_from_hdf, save_csr_to_hdf
from settings import K_MAX

SYNTHETIC_CONFIGURATIONS = (
    (2, 12),
    (2, 15),
    (4, 15),
    (8, 15)
)
ROAD_NETWORKS = (
    'OL',
    'cal',
    'TG',
    'SF',
)
TOTAL_NUMBER_OF_DATASETS = len(SYNTHETIC_CONFIGURATIONS) + len(ROAD_NETWORKS)

HDF_SETTINGS = {
    'compression': 'gzip',
    'compression_opts': 9,
    'shuffle': True,
    'fletcher32': True,
    'chunks': True,
}


def load_road_networks(
        dataset_name: str = 'OL',
        root: str = '/mnt/data/road_networks',
        shuffle: bool = True,
        seed: int = 42
) -> numpy.ndarray:
    """
    Load a road network dataset downloaded from https://www.cs.utah.edu/~lifeifei/SpatialDataset.htm.

    :param dataset_name: str
        The name of the dataset. See ROAD_NETWORKS for allowed keys.
    :param root: str
        The data root.
    :param shuffle: bool
        Whether to shuffle.
    :param seed: int
        The seed for shuffling.

    :return: ndarray, shape: (n, 2), dtype: float32
        The array of data points.
    """
    file_name = '{n}.cnode.txt'.format(n=dataset_name)
    file_path = path.join(root, file_name)
    df = pandas.read_csv(file_path, sep=' ', header=None, names=['Node ID', 'x', 'y'])
    values = df.loc[:, ['x', 'y']].values.astype(numpy.float32)
    if shuffle:
        numpy.random.seed(seed=seed)
        numpy.random.shuffle(values)
    return values


def process_data_set(
        x: numpy.ndarray,
        dataset_name: str,
        index_root: str,
        model_root: str,
        overwrite: bool = False,
        n_jobs: int = -1
) -> None:
    """
    Preprocess a dataset.

    :param x: numpy.ndarray, shape: (n, d), dtype: float
        The data.
    :param dataset_name: str
        The name.
    :param index_root: str
        The root directory under which to store the index files.
    :param model_root: str
        The root directory under which to store the model files.
    :param overwrite: bool
        Whether to overwrite existing files.
    :param n_jobs: int
        The number of jobs to use for computing the k-distance.

    :return: None
    """
    # Compose index file path
    index_file_name = '{n}.index'.format(n=dataset_name)
    index_file_path = path.join(index_root, index_file_name)

    # Check if something needs to be done.
    if not path.isfile(index_file_path) or overwrite:
        # Create BallTree index
        tree = BallTree(x)

        # Save using pickle
        with open(index_file_path, 'wb') as index_file:
            pickle.dump(tree, index_file, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        logging.info(f'Skipping to overwrite existing BallTree index file: {index_file_path}')

    # Create HDF5 file
    # Compose k-distance file path
    h5_file_name = '{n}.h5'.format(n=dataset_name)
    h5_file_path = path.join(index_root, h5_file_name)

    knn_graph = None
    # Check if something needs to be done.
    if not path.isfile(h5_file_path) or overwrite:
        with h5py.File(h5_file_path, mode='w', libver='latest') as h5f:
            # Create knn distance graph
            knn_graph = kneighbors_graph(x, n_neighbors=K_MAX, mode='distance', n_jobs=n_jobs)

            # Save knn distances
            save_csr_to_hdf(knn_graph, h5f=h5f, key='distances', **HDF_SETTINGS)
    else:
        logging.info(f'Skipping to overwrite existing k-distance file: {h5_file_path}')

    mrknn_file_name = f'{dataset_name}.mrknn'
    mrknn_path = path.join(model_root, mrknn_file_name)

    # Check if something needs to be done.
    if not path.isfile(mrknn_path) or overwrite:
        # Load k-distances if necessary
        if knn_graph is None:
            with h5py.File(h5_file_path, mode='r', libver='latest') as h5f:
                knn_graph = load_csr_from_hdf(h5f=h5f, key='distances')

        # Sort distances
        kd = knn_graph.data.reshape([knn_graph.shape[0], -1])
        skd = numpy.sort(kd, axis=-1)

        # Fit MRkNNCoP tree
        logging.info('Training MRkNNCoP tree')
        mrknn = MRkNNCoPTree()
        mrknn.fit(y=skd)

        logging.info(f'Saving model to {mrknn_path}')
        with open(mrknn_path, 'wb') as pf:
            pickle.dump(mrknn, pf, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        logging.info(f'Skipping to overwrite existing file: {mrknn_path}')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(asctime)s %(message)s')

    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--index_root', type=str, default='./index', help='The index root.')
    parser.add_argument('--model_root', type=str, default='./models', help='The models root.')
    args = parser.parse_args()

    # Create index root, if it does not exist
    os.makedirs(args.index_root, exist_ok=True)
    os.makedirs(args.model_root, exist_ok=True)

    # Synthetic datasets
    for i, (d, e) in enumerate(SYNTHETIC_CONFIGURATIONS):
        logging.info(f'[{i + 1:2d}/{TOTAL_NUMBER_OF_DATASETS:2d}] Processing make_blobs(n={2 ** e}, d={d})')
        n = 2 ** e
        x, y = make_blobs(n_samples=n, n_features=d, centers=e)
        process_data_set(x=x, dataset_name=f'blobs.{d}.{e}', index_root=args.index_root, overwrite=False, model_root=args.model_root)

    # Road networks
    for i, name in enumerate(ROAD_NETWORKS, start=len(SYNTHETIC_CONFIGURATIONS)):
        logging.info(f'[{i + 1:2d}/{TOTAL_NUMBER_OF_DATASETS:2d}] Processing road network {name}.')
        x = load_road_networks(dataset_name=name, shuffle=True, seed=42)
        process_data_set(x=x, dataset_name=name, index_root=args.index_root, overwrite=False, model_root=args.model_root)
