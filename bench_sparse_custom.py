import pandas as pd
import xarray as xr
import sparse as pydata_sparse
from scipy import sparse

from sklearn import set_config
from sklearn.utils._data_adapter import _DataTransformer
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array

from bench_this import Benchmark


class SillyVectorizer(TransformerMixin, BaseEstimator):
    def __init__(self, n_features_out=1_000, density=0.01):
        self.n_features_out = n_features_out
        self.density = density

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        data_wrap = _DataTransformer(X, needs_feature_names_in=False)

        n_samples = len(X)
        X_output = sparse.rand(n_samples, self.n_features_out,
                               density=self.density, random_state=0)
        output = data_wrap.transform(X_output, self.get_feature_names)
        return output

    def get_feature_names(self):
        return [f'col_{i}' for i in range(self.n_features_out)]


class PassthroughTransformer(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        X = check_array(X, accept_sparse=True)
        # do some fitting
        return self

    def transform(self, X):
        data_wrap = _DataTransformer(X)
        X = check_array(X, accept_sparse=True)
        # typically does some math
        return data_wrap.transform(X)


class SparseTextBenchmark(Benchmark):
    _PARAM_DICT = {
        'array_out': ['default', 'xarray', 'pandas', 'pydata/sparse'],
        'density': [0.1, 0.2, 0.3, 0.4, 0.5]
    }
    _REPEAT = 1

    def single(self, *, density, array_out):
        set_config(array_out=array_out)

        n_samples = 100_000
        X = [None] * n_samples

        pipe = make_pipeline(SillyVectorizer(density=density),
                             PassthroughTransformer())
        pipe.fit(X)
        output = pipe.transform(X)

        # sanity check
        if array_out == 'pandas':
            assert isinstance(output, pd.DataFrame)
        elif array_out == 'xarray':
            assert isinstance(output, xr.DataArray)
        elif array_out == 'pydata/sparse':
            assert isinstance(output, pydata_sparse.COO)
        else:  # default
            assert sparse.issparse(output)


if __name__ == '__main__':
    SparseTextBenchmark._cli()
