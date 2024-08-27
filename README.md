# Higher-order interactions improve the prediction accuracy of reservoir computer

This repository contains the code used for paper "Higher-order interactions
improve the prediction accuracy of reservoir computer".

## Project structure

Project root folder contains Jupyter notebooks used for recreating corresponding figures from the paper.

- `/Data`: Training and testing data for reservoir computer, including FitzHugh-Nagumo, Lorenz, Rössler time series,
as well as EEG recording data.
- `/Methods`: Reservoir computer models, configuration file and scripts for running models.
Each of the `run_esn_figN_*.py` scripts initializes RC training and prediction to obtain data used for
plotting Figure N from the paper.
- `/Results`: Evaluated entropic characteristics data and RC prediction data used for plotting. Also contains log files
and trained models.

## Dependencies
- `python 3.10.7`
- `matplotlib 3.6.0`
- `numpy 1.22.4`
- `scikit-learn 1.1.2`
- `scipy 1.8.1`
