import sys
import numpy as np
from RUN import runModel
from Config.global_conf import global_params

optcfg = {
    "exp": "fhn_d005-o3n-monte_carlo_lorenz300n_dt0.01-35/",
}


if __name__ == "__main__":
    global_params.training_data_path = \
        global_params.project_path + "/Data/rcc_train_lorenz_19.04.24-2-dt0.01.pickle"
    global_params.testing_data_path = \
        global_params.project_path + "/Data/rcc_test_lorenz_19.04.24-2-dt0.01.pickle"
    args_dict = {
        "model_name": "esn_o3_chaos",
        "mode": "all",
        "display_output": 0,
        "system_name": "Lorenz3D",
        "write_to_log": 1,
        "N_used": 10000,
        "RDIM": 3,
        "noise_level": 0,
        "scaler": "Standard",
        "approx_reservoir_size": 300,
        "sigma_input": 1,
        "regularization": 0.0001,
        "dynamics_length": 2000,
        "iterative_prediction_length": 6000,
        "num_test_ICS": 1,
        "solver": "auto",
        "number_of_epochs": 1000000,
        "learning_rate": 0.001,
        "reference_train_time": 10,
        "buffer_train_time": 0.5,
        
        "degree": 10,
        "radius": 1.0,
        "worker_id": 136,
        "alpha_1": 0.0,
        "alpha_2": 1.0,
        "w_in_seed": 35,
    }

    args_dict["saving_path"] = global_params.saving_path

    args_dict["model_dir"] = global_params.model_dir + optcfg['exp']
    args_dict["fig_dir"] = global_params.fig_dir + optcfg['exp']
    args_dict["results_dir"] = global_params.results_dir + optcfg['exp']
    args_dict["logfile_dir"] = global_params.logfile_dir + optcfg['exp']

    args_dict["train_data_path"] = global_params.training_data_path
    args_dict["test_data_path"] = global_params.testing_data_path

    runModel(args_dict)