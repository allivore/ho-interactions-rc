import sys
from Config.global_conf import global_params
sys.path.insert(0, global_params.global_utils_path)


def getModel(params):
	sys.path.insert(0, global_params.py_models_path.format(params["model_name"]))
	if params["model_name"] == "esn_o1":
		import esn_o1 as model
		return model.esn_o1(params)
	elif params["model_name"] == "esn_o2":
		import esn_o2 as model
		return model.esn_o2(params)
	elif params["model_name"] == "esn_o3":
		import esn_o3 as model
		return model.esn_o3(params)
	elif params["model_name"] == "esn_o3_nodeg":
		import esn_o3_nodeg as model
		return model.esn_o3_nodeg(params)
	elif params["model_name"] == "esn_o3_58_cliques_best_clique":
		import esn_o3_58_cliques_best_clique as model
		return model.esn_o3_58_cliques_best_clique(params)
	elif params["model_name"] == "esn_o3_12_cliques_best_clique":
		import esn_o3_12_cliques_best_clique as model
		return model.esn_o3_12_cliques_best_clique(params)
	elif params["model_name"] == "esn_o3_chaos":
		import esn_o3_chaos as model
		return model.esn_o3_chaos(params)
	elif params["model_name"] == "esn_o3_eeg":
		import esn_o3_eeg as model
		return model.esn_o3_eeg(params)
	else:
		raise ValueError("model not found.")

def runModel(params_dict):
	if params_dict["mode"] in ["train", "all"]:
		trainModel(params_dict)
	if params_dict["mode"] in ["test", "all"]:
		testModel(params_dict)
	return 0

def trainModel(params_dict):
	model = getModel(params_dict)
	model.train()
	model.delete()
	del model
	return 0

def testModel(params_dict):
	model = getModel(params_dict)
	model.testing()
	model.delete()
	del model
	return 0

