from subprocess import run
from itertools import product
import json

PARAM_DICT = {
    'max_features': [10000, 50000, 100000, 130107],
    'array_out': ['default', 'pandas', 'xarray'],
}
REPEAT = 1

params = product(*list(PARAM_DICT.values()))
params = (dict(zip(PARAM_DICT, param)) for param in params)

results = []
for param in params:
    for i in range(REPEAT):
        cmd = ["python", "bench_newsgroup.py"]
        for k, v in param.items():
            cmd.extend([f"--{k}", f"{v}"])

        output = run(cmd, capture_output=True)
        result = json.loads(output.stdout)
        print(result)
        results.append(result)

with open("benchmark_newsgroup.json", 'w') as f:
    json.dump(results, f)
