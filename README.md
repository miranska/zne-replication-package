# Improving Zero-Noise Extrapolation via Physically Bounded Models
![CI status badge](https://github.com/miranska/zne-replication-package/actions/workflows/pytest.yml/badge.svg?branch=main)
[![Coverage badge](https://github.com/miranska/zne-replication-package/raw/python-coverage-comment-action-data/badge.svg)](https://github.com/miranska/zne-replication-package/tree/python-coverage-comment-action-data)
![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)

This repository contains supplementary material for the paper 
"Improving Zero-Noise Extrapolation via Physically Bounded Models".

## Installation

This project uses `pyproject.toml` for dependency management.
Use Python `3.11` or `3.12`.

With `venv` and `pip`:

1. Create and activate a virtual environment:
   ```shell
   python -m venv .venv
   source .venv/bin/activate
   ```
2. Install the package and runtime dependencies:
   ```shell
   pip install -e .
   ```
3. (Optional) Install test dependencies:
   ```shell
   pip install -e ".[test]"
   ```

With `uv`:

1. Sync the runtime environment:
   ```shell
   uv sync
   ```
2. (Optional) Sync test dependencies:
   ```shell
   uv sync --extra test
   ```
3. Activate the environment if needed:
   ```shell
   source .venv/bin/activate
   ```

## Repository Structure

### `src/`

- `bounded_methods.py`: implementations of bounded extrapolation routines for zero-noise estimation:
  - `bounded_polynomial_extrapolation`: constrained polynomial model with bounded intercept.
  - `bounded_exp_extrapolation`: constrained exponential model.
  - `bounded_polyexp_extrapolation`: constrained polynomial--exponential model.

### `tests/`
- `test_bounded_methods.py`: unit tests for `bounded_methods.py`.

### `notebooks/`

- `00_demo_zne_methods.ipynb`: demonstrates ZNE methods and example analysis flow.
- `01_access_qasm.ipynb`: shows how to access and inspect circuit files in QASM archives.
- `02_access_measurements.ipynb`: shows how to read synthetic and real-device measurement records from compressed JSONL files.
- `03_access_evaluations.ipynb`: shows how to access evaluation result archives (synthetic and real-device) stored as parquet datasets.

## Data Setup (Required for notebooks `01_`, `02_`, `03_`)

Notebooks with prefixes greater than `00_` require local data files under `data/`.

1. Download the required archives/files from [Zenodo](https://doi.org/10.5281/zenodo.19712079) using your preferred method (web browser, `wget`, `curl`, etc.).
2. Place all downloaded files into the `data/` directory.
3. Unpack the zip archives listed below inside `data/`:
   - `evaluation_archive.zip`
   - `evaluation_real_archive.zip`
   
   For example, using:
    ```shell
    unzip data/evaluation_archive.zip -d data/evaluation_archive
    unzip data/evaluation_real_archive.zip -d data/evaluation_real_archive
    ```

## Testing
Run unit tests using:
```shell
pytest
```

## Citation
If you use or study the code, please cite it as follows.
