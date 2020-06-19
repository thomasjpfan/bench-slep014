import pandas as pd
import xarray as xr
import numpy as np

from sklearn import set_config

from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.feature_selection import SelectKBest
from sklearn.datasets import fetch_openml

from bench_this import Benchmark


class DenseBenchmark(Benchmark):
    _PARAM_DICT = {
        'array_out': ['default', 'pandas', 'xarray'],
    }
    _REPEAT = 5

    def single(self, *, array_out):
        X, y = fetch_openml(data_id=1476, return_X_y=True, as_frame=True)

        set_config(array_out=array_out)
        pipe = make_pipeline(StandardScaler(),
                             PCA(n_components=64),
                             SelectKBest(k=30),
                             Ridge())
        pipe.fit(X, y)
        output = pipe[:-1].transform(X)

        # sanity check
        if array_out == 'pandas':
            assert isinstance(output, pd.DataFrame)
        elif array_out == 'xarray':
            assert isinstance(output, xr.DataArray)
        else:  # default
            assert isinstance(output, np.ndarray)

    def download(self):
        """Download data"""
        print("Downloading openml dataset 1476")
        fetch_openml(data_id=1476)


if __name__ == '__main__':
    DenseBenchmark._cli()
