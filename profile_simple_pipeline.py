import numpy as np
from sklearn.utils._data_adapter import _DataTransformer
from sklearn.datasets import make_classification
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.utils.validation import check_array


class PassthroughTransformer(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        data_wrap = _DataTransformer(X)
        X = check_array(X)
        # typically does some math
        return data_wrap.transform(X)


class SillyClassifer(ClassifierMixin, BaseEstimator):
    def fit(self, X, y):
        return self

    def predict(self, X):
        X = check_array(X)
        return np.ones(X.shape[0], dtype=int)


X, y = make_classification(n_samples=500_000, n_features=100, random_state=4)

pipe = make_pipeline(PassthroughTransformer(), SillyClassifer())

pipe.fit(X, y)
pipe.predict(X)
