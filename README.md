# Benchmaking SLEP 014

1. Install requirements.

```bash
pip install -r requirements.txt
```

2. Install [PR #16772](https://github.com/scikit-learn/scikit-learn/pull/16772) from
source.

## Text input that generates sparse data

**Note** Before running the script for the first time, you can download the
dataset by running `python bench_sparse_text_input.py download`.

To benchmark run:

```py
python bench_sparse_text_input.py full
```

## Dense dataset

**Note** Before running the script for the first time, you can download the
dataset by running `python bench_dense.py download`.

```py
python bench_dense.py full
```
