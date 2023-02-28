# MLE module 3
This project build models to predict mean housing value.

## Table of Contents
    - Requirements
    - Configuration and Installation
    - Running python scripts

## Requirements
No special requirements.

## Configuration and Installation
To install this project, follow these steps:
1. Clont the repository in your local machine.
```git clone https://github.com/reeshabhc/mle-training/blob/fix/16/mle_module_3```
2. Navigate to the project directory.
```cd project```
3. Install the required dependencies.
```conda env create --f deploy/conda/mle3_dev_env.yml```
4. Set up any necessary configurations.
```python setup.py sdist```
```python setup.py build```
5. To install the package
```pip install .```
6. To uninstall the package
```pip uninstall MLE_Module_3```

## Running python scripts
To run the files, move to the root directory:
1. Run ingest_data.py
```python --output <optional_output_path> src/ingest_data.py``` Other optional arguments --log-level, --log_path, --no-console-log
2. Run train.py
```python --inp_train --out_pkl --out_score src/train.py``` Other optional arguments --log-level, --log_path, --no-console-log
3. score.py
```python --model_dir --test_data_dir --output_dir src/train.py``` Other optional arguments --log-level, --log_path, --no-console-log
