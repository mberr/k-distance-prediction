# coding=utf-8
"""
Persistence utilities.
"""
import h5py
from scipy import sparse


def save_csr_to_hdf(
        csr_matrix: sparse.csr_matrix,
        h5f: h5py.File,
        key: str,
        **kwargs
) -> None:
    """
    Save a CSR matrix to already opened HDF5 file.

    :param csr_matrix: sparse.csr_matrix
        The matrix.
    :param h5f: h5py.File
        The opened HDF5 file.
    :param key: str
        A key.
    :param kwargs: Dict[str, Any]
        Additional settings passed to h5py.create_dataset

    :return: None.
    """
    # Read out components of CSR matrix
    data, indices, indptr = csr_matrix.data, csr_matrix.indices, csr_matrix.indptr

    # Create group
    h5g = h5f.create_group(name=key)

    # Create dataset per component
    h5g.create_dataset(name='data', data=data, **kwargs)
    h5g.create_dataset(name='indices', data=indices, **kwargs)
    h5g.create_dataset(name='indptr', data=indptr, **kwargs)


def load_csr_from_hdf(
        h5f: h5py.File,
        key: str
) -> sparse.csr_matrix:
    """
    Load a CSR file from already opened HDF5 file.

    :param h5f: h5py.File
        The already opened file.
    :param key: str
        The key.

    :return: sparse.csr_matrix
        The recovered CSR matrix.
    """
    # Find group
    group = h5f[key]

    # Read out components
    data = group['data']
    indices = group['indices']
    indptr = group['indptr']

    # Compose matrix
    matrix = sparse.csr_matrix((data, indices, indptr))

    return matrix
