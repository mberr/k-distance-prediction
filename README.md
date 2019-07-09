# k-Distance Approximation for Memory-Efficient RkNN Retrieval
Repository containing the code for the paper

__k-Distance Approximation for Memory-Efficient RkNN Retrieval__  
_Max Berrendorf, Felix Borutta, and Peer Kröger_  
SISAP'19

If you find this code useful, please consider citing us.
```
@InProceedings{kDistanceApproximation,
    author="Berrendorf, Max
    and Borutta, Felix
    and Kr{\"o}ger, Peer",
    title="k-Distance Approximation for Memory-Efficient RkNN Retrieval",
    booktitle="Similarity Search and Applications",
    year="2019",
    publisher="Springer International Publishing",
}

```

## Data Preprocessing
Download road networks `OL`, `TG`, and `SF` from [here](https://www.cs.utah.edu/~lifeifei/SpatialDataset.htm).
Then, run preprocessing script to
* generate synthetic datasets. 
* for each dataset, compute and store
  * `BallTree` index for fast range queries in evaluation code, and 
  * `k-distances` for building the index.
  * `MRkNNCoP` tree coefficients.
You can use the command line argument `--index_root` to specify a directory where the data is stored (consuming approx. `1.1 GiB`), and `--model_root` to specify a directory to store the MRkNNCoP tree coefficients (consuming approx. `5.0 MiB`).
```bash
python3 preprocess.py --index_root=./index --model_root=./models
```
The indices are stored as `pickle` file named `<dataset_name>.pkl`, and the k-distances as sparse CSR matrices in a `HDF5` file in a custom format (cf. `persistence.py:save_csr_to_hdf`).


## Model Selection
After preprocessing you can perform model selection to analyse the trade-off between model size and candidate set size.
To this end, use the following
```bash
python3 model_selection.py --index_root=./index --model_root=./models --output_root=./results
```
which reads data and MRkNNCoP tree coefficients and trains numerous predefined models to predict the MRkNNCoP tree coefficients and thereby the k-distance.
As a result, for each of the datasets the following files are produced:
* a CSV containing the error in coefficients prediction for all models
* a CSV containing the mean candidate set size for every model in the `model_size`-`mae` skyline.
* a pickle file containing the candidate set sizes for each model for each data point and each value of k between 1 and K_MAX

The output files consume approx. `4.0 GiB`.    

## Evaluation
After training all models, the `notebooks/Evaluation.ipynb` can be used to evaluate the results.
