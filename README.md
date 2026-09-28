# Machine Learning Assignment 3: Incremental Class Learning

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running

All scripts are run from the project root.

```bash
python src/download_data.py                        # downloads the datasets
python src/tune_hidden.py                          # tunes the baseline layer size
python src/tune_icl.py                             # tunes k for ICL
python src/run_experiments.py baseline covertype
python src/run_experiments.py icl covertype
python src/analyse_results.py
```

`run_experiments.py` takes a method (`baseline` or `icl`) and a dataset
(`wall_robot`, `cardiotocography` or `covertype`). Run it for every
combination before `analyse_results.py`.

## Structure

```
data/raw/                   the three datasets, unmodified
src/
  config.py
  preprocessing.py          loading and cleaning of the data
  splits.py
  network.py                the neural network
  optimisers.py
  training.py
  experiment.py
  icl.py                    the incremental class learning algorithm
  tune_hidden.py
  tune_icl.py
  run_experiments.py
  analyse_results.py
  download_data.py          downloads the datasets
  data_analysis.ipynb       data exploration
results/
  tuning/                   tuning results
  analysis/                 result tables as csv
  figures/                  figures
requirements.txt
```
