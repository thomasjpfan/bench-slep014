import pandas as pd
import xarray as xr
from scipy import sparse

from sklearn.datasets import fetch_20newsgroups
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MaxAbsScaler
from sklearn import set_config

from bench_this import Benchmark


class SparseTextBenchmark(Benchmark):
    _PARAM_DICT = {
        'array_out': ['default', 'pandas', 'xarray'],
        'maxabs_scalers': [1, 2, 3]
    }
    _REPEAT = 5

    def single(self, *, maxabs_scalers, array_out):
        data = fetch_20newsgroups(subset='train')
        set_config(array_out=array_out)

        estimators = ([CountVectorizer()] +
                      [MaxAbsScaler() for _ in range(maxabs_scalers)])
        pipe = make_pipeline(*estimators)
        output = pipe.fit_transform(data.data)

        # sanity check
        if array_out == 'pandas':
            assert isinstance(output, pd.DataFrame)
        elif array_out == 'xarray':
            assert isinstance(output, xr.DataArray)
        else:  # default
            assert sparse.issparse(output)

    def download(self):
        """Download data"""
        print("Downloading 20newsgroup")
        fetch_20newsgroups(subset='train')


if __name__ == '__main__':
    SparseTextBenchmark._cli()
