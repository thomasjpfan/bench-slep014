import pandas as pd
import xarray as xr
import numpy as np

from sklearn import set_config

from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.compose import make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


from bench_this import Benchmark


class ColumnTransformBenchmark(Benchmark):

    _PARAM_DICT = {
        'array_out': ['default', 'pandas', 'xarray'],
    }
    _REPEAT = 5

    def single(self, *, array_out):
        X, y = fetch_openml(data_id=1590, return_X_y=True, as_frame=True)

        set_config(array_out=array_out)
        cat_prep = make_pipeline(
            SimpleImputer(fill_value='sk_missing',
                          strategy='constant'),
            OneHotEncoder(handle_unknown='ignore', sparse=False)
        )

        prep = make_column_transformer(
            (StandardScaler(), make_column_selector(dtype_include='number')),
            (cat_prep, make_column_selector(dtype_include='category'))
        )

        pipe = make_pipeline(prep,
                             SelectKBest(),
                             DecisionTreeClassifier(random_state=42))
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
        print("Downloading openml dataset 1590")
        fetch_openml(data_id=1590)


if __name__ == '__main__':
    ColumnTransformBenchmark._cli()
