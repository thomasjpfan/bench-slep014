from sklearn.datasets import fetch_20newsgroups
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import set_config

from bench_this import Benchmark


class SparseTextBenchmark(Benchmark):
    _PARAM_DICT = {
        'max_features': [10000, 50000, 100000, 130107],
        'array_out': ['default', 'pandas', 'xarray', 'pydata/sparse'],
    }
    _REPEAT = 5

    def single(self, *, max_features, array_out):
        data = fetch_20newsgroups(subset='train')
        set_config(array_out=array_out)
        pipe = make_pipeline(CountVectorizer(max_features=max_features),
                             TfidfTransformer(),
                             SGDClassifier(random_state=42))
        pipe.fit(data.data, data.target)

    def download(self):
        """Download data"""
        print("Downloading 20newsgroup")
        fetch_20newsgroups(subset='train')


if __name__ == '__main__':
    SparseTextBenchmark._cli()
