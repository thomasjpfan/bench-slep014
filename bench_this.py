import csv
import inspect
from abc import ABC, abstractmethod
from pathlib import Path
from subprocess import run
from functools import wraps
from time import perf_counter_ns
from itertools import product
import json
import sys

import fire
from memory_profiler import memory_usage


def benchmark(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        start = perf_counter_ns()
        usage = memory_usage((f, args, kwargs), interval=0.001)
        duration = (perf_counter_ns() - start) / 1E9
        usage = [el - usage[0] for el in usage]
        kwargs.update({"peak_memory": max(usage), "time": duration})
        print(json.dumps(kwargs))
    return wrapper


class Benchmark(ABC):
    # private to hide from fire.Fire
    _PARAM_DICT = None
    _REPEAT = None

    def __init__(self, overwrite=False):
        self.overwrite = overwrite

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        assert cls._PARAM_DICT is not None
        assert cls._REPEAT is not None

        # check that param_dict has keys that match cls.single
        assert (set(cls._PARAM_DICT) <=
                set(inspect.signature(cls.single).parameters))

        assert cls._REPEAT > 0

        # wraps script to benchmark
        cls.single = benchmark(cls.single)

    @abstractmethod
    def single(self):
        """Single benchmark run"""

    def full(self):
        """Run full benchmark with param_dict and repeat"""
        file_name = inspect.getfile(self.__class__)

        output = Path("results") / f"{Path(file_name).stem}.csv"
        if output.exists() and not self.overwrite:
            print(f"{output} exists, please pass --overwrite=True to "
                  "overwrite file")
            sys.exit(1)

        params = product(*self._PARAM_DICT.values())
        params = (dict(zip(self._PARAM_DICT, param)) for param in params)

        keys = list(self._PARAM_DICT) + ['peak_memory', 'time']
        with output.open('w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=keys)
            writer.writeheader()

            for param in params:
                for i in range(self._REPEAT):
                    cmd = ["python", file_name, "single"]
                    for k, v in param.items():
                        cmd.extend([f"--{k}", f"{v}"])

                    output = run(cmd, capture_output=True)
                    result = json.loads(output.stdout)
                    print(result)
                    writer.writerow(result)
                    csvfile.flush()

    @classmethod
    def _cli(cls):
        fire.Fire(cls)
