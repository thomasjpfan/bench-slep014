from time import perf_counter_ns
from memory_profiler import memory_usage
import fire
from functools import wraps
import json

from sklearn.datasets import fetch_20newsgroups
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import config_context


def benchmark(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        start = perf_counter_ns()
        usage = memory_usage((f, args, kwargs), interval=0.01)
        duration = (perf_counter_ns() - start) / 1E9
        usage = [el - usage[0] for el in usage]
        kwargs.update({"peak_memory": max(usage), "time": duration})
        print(json.dumps(kwargs))

    return wrapper


data = fetch_20newsgroups(subset='train')


@benchmark
def _run(*, max_features, array_out):
    with config_context(array_out=array_out):
        pipe = make_pipeline(CountVectorizer(max_features=max_features),
                             TfidfTransformer(),
                             SGDClassifier(random_state=42))
        pipe.fit(data.data, data.target)


if __name__ == '__main__':
    fire.Fire(_run)
