import numpy as np
from sklearn.utils._array_transformer import _ArrayTransformer
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array
from sklearn import set_config
import scipy.sparse as sparse
import pandas as pd

from bench_this import Benchmark


def _generate_csr_matrix1(n_samples, n_features):
    rng = np.random.RandomState(42)
    row = rng.randint(0, n_samples, 10 * n_samples)
    col = rng.randint(0, n_features, 10 * n_samples)
    entry = rng.normal(size=10 * n_samples)
    X_output = sparse.coo_matrix((entry, (row, col)),
                                 shape=(n_samples, n_features)).tocsr()
    return X_output


class SillyVectorizer(TransformerMixin, BaseEstimator):
    def __init__(self, n_features=100_000):
        self.n_features = n_features

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        data_wrap = _ArrayTransformer(X, needs_feature_names_in=False)
        n_samples = len(X)
        X_output = _generate_csr_matrix1(n_samples, self.n_features)

        return data_wrap.transform(X_output, self.get_feature_names)

    def get_feature_names(self):
        return [f'col_{i}' for i in range(self.n_features)]


class PassthroughTransformer(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        X = check_array(X, accept_sparse=True)
        # do some fitting
        return self

    def transform(self, X):
        data_wrap = _ArrayTransformer(X)
        X = check_array(X, accept_sparse=True)
        # typically does some math
        return data_wrap.transform(X)


class SimpleSparseBenchmark(Benchmark):

    _PARAM_DICT = {
        'array_out': ['default', 'pandas', 'xarray', 'pydata/sparse',
                      'pandas/pydata/sparse'],
        'n_features': [100_000, 300_000, 500_000],
        'n_passthrough': [1, 2, 3]
    }
    _REPEAT = 5

    def single(self, *, array_out, n_features, n_passthrough):
        set_config(array_out=array_out)

        X = ["text"] * 1_000_000

        pipe_est = ([SillyVectorizer(n_features=n_features)] +
                    [PassthroughTransformer() for _ in range(n_passthrough)])

        pipe = make_pipeline(*pipe_est)
        pipe.fit(X)
        output = pipe.transform(X)

        # sanity check
        if array_out == 'pandas':
            assert isinstance(output, pd.DataFrame)
        elif array_out == 'xarray':
            import xarray as xr
            assert isinstance(output, xr.DataArray)
        elif array_out == 'pydata/sparse':
            import sparse as pydata_sparse
            assert isinstance(output, pydata_sparse.COO)
        elif array_out == 'pandas/pydata/sparse':
            import sparse as pydata_sparse
            assert isinstance(output, pd.DataFrame)
            assert isinstance(output.data, pydata_sparse.COO)
        else:  # default
            assert sparse.issparse(output)


if __name__ == '__main__':
    SimpleSparseBenchmark._cli()
