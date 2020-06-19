import pandas as pd
import xarray as xr
import numpy as np

from sklearn.datasets import make_regression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn import set_config

from bench_this import Benchmark


class DenseMinMaxBench(Benchmark):

    _PARAM_DICT = {
        'array_out': ['default', 'pandas', 'xarray'],
        'minmax_scalers': [1, 2, 3, 4, 5]
    }
    _REPEAT = 5

    def single(self, *, array_out, minmax_scalers):
        n_features = 200
        X, _ = make_regression(n_samples=300_000, n_features=n_features,
                               random_state=42)
        df = pd.DataFrame(X, columns=[f"col_{i}" for i in range(n_features)])
        set_config(array_out=array_out)

        pipe = make_pipeline(*[MinMaxScaler() for _ in range(minmax_scalers)])
        output = pipe.fit_transform(df)

        # sanity check
        if array_out == 'pandas':
            assert isinstance(output, pd.DataFrame)
        elif array_out == 'xarray':
            assert isinstance(output, xr.DataArray)
        else:  # default
            assert isinstance(output, np.ndarray)


if __name__ == '__main__':
    DenseMinMaxBench._cli()
