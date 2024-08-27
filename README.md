# Higher-order interactions improve the prediction accuracy of reservoir computer

This repository contains the code used for paper "Higher-order interactions
improve the prediction accuracy of reservoir computer".

## Project structure

Project root folder contains Jupyter notebooks used for recreating corresponding figures from the paper.

- [`/Data`](./Data): Training and testing data for reservoir computer, including FitzHugh-Nagumo, Lorenz, Rössler time series,
as well as EEG recording data.
- [`/Methods`](./Methods): [Reservoir computer models](./Methods/Models), [configuration file](./Methods/Config) and scripts for running models.
Each of the `run_esn_figN_*.py` scripts initializes RC training and prediction to obtain data used for
plotting Figure N from the paper.
- [`/Results`](./Results): [Evaluated entropic characteristics data](./Results/Entropy) and [RC prediction data](./Results/Evaluation_Data) used for plotting. Also contains [log files](./Results/Logfiles) and [trained models](./Results/Trained_Models).

## Dependencies
- `python 3.10.7`
- `matplotlib 3.6.0`
- `numpy 1.22.4`
- `scikit-learn 1.1.2`
- `scipy 1.8.1`
