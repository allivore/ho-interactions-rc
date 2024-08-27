import sys
import numpy as np
from RUN import runModel
from Config.global_conf import global_params

optcfg = {
    "exp": "fhn_d005-o2n-a=1.0n0.0-r500-opt-seed=7/",
}


if __name__ == "__main__":
    global_params.training_data_path = \
        global_params.project_path + "/Data/rcc_train_fhn_new_d005scaled_3ch_seed2001.pickle"
    global_params.testing_data_path = \
        global_params.project_path + "/Data/rcc_test_fhn_new_d005scaled_3ch_seed2002.pickle"
    args_dict = {
        "model_name": "esn_o2",
        "mode": "all",
        "display_output": 0,
        "system_name": "Lorenz3D",
        "write_to_log": 1,
        "N": 100000,
        "N_used": 30000,
        "RDIM": 3,
        "noise_level": 0,
        "scaler": "Standard",
        "approx_reservoir_size": 500,
        "sigma_input": 1,
        "regularization": 0.0001,
        "dynamics_length": 10000,
        "iterative_prediction_length": 20000,
        "num_test_ICS": 1,
        "solver": "auto",
        "number_of_epochs": 1000000,
        "learning_rate": 0.001,
        "reference_train_time": 10,
        "buffer_train_time": 0.5,
        
        "degree": 14,
        "radius": 1.6,
        "worker_id": 7,
        "alpha_1": 1.0,
        "alpha_2": 0.0,
    }

    args_dict["saving_path"] = global_params.saving_path

    args_dict["model_dir"] = global_params.model_dir + optcfg['exp']
    args_dict["fig_dir"] = global_params.fig_dir + optcfg['exp']
    args_dict["results_dir"] = global_params.results_dir + optcfg['exp']
    args_dict["logfile_dir"] = global_params.logfile_dir + optcfg['exp']

    args_dict["train_data_path"] = global_params.training_data_path
    args_dict["test_data_path"] = global_params.testing_data_path

    runModel(args_dict)