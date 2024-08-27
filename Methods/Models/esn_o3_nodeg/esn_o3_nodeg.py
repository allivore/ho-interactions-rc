import numpy as np
import pickle
from scipy import sparse as sparse
from scipy.sparse import linalg as splinalg
from scipy.linalg import pinv2 as scipypinv2
import os
from plotting_utils import *
from global_utils import *
import pickle
import time
from functools import partial
print = partial(print, flush=True)

from sklearn.linear_model import Ridge
from sklearn.preprocessing import minmax_scale

import psutil

class esn_o3_nodeg(object):
	def delete(self):
		return 0
		
	def __init__(self, params):
		self.display_output = params["display_output"]

		print("RANDOM SEED: {:}".format(params["worker_id"]))
		np.random.seed(params["worker_id"])

		self.worker_id = params["worker_id"]
		self.input_dim = params["RDIM"]
		self.N_used = params["N_used"]
		self.approx_reservoir_size = params["approx_reservoir_size"]
		self.degree = params["degree"]
		self.radius = params["radius"]
		self.sigma_input = params["sigma_input"]
		self.dynamics_length = params["dynamics_length"]
		self.iterative_prediction_length = params["iterative_prediction_length"]
		self.num_test_ICS = params["num_test_ICS"]
		self.train_data_path = params["train_data_path"]
		self.test_data_path = params["test_data_path"]
		self.fig_dir = params["fig_dir"]
		self.model_dir = params["model_dir"]
		self.logfile_dir = params["logfile_dir"]
		self.write_to_log = params["write_to_log"]
		self.results_dir = params["results_dir"]
		self.saving_path = params["saving_path"]
		self.regularization = params["regularization"]
		self.scaler_tt = params["scaler"]
		self.learning_rate = params["learning_rate"]
		self.number_of_epochs = params["number_of_epochs"]
		self.solver = str(params["solver"])
		self.alpha_1 = params["alpha_1"]
		self.alpha_2 = params["alpha_2"]
		##########################################
		self.scaler = scaler(self.scaler_tt)
		self.noise_level = params["noise_level"]
		self.model_name = self.createModelName(params)

		self.reference_train_time = 60*60*(params["reference_train_time"]-params["buffer_train_time"])
		print("Reference train time {:} seconds / {:} minutes / {:} hours.".format(self.reference_train_time, self.reference_train_time/60, self.reference_train_time/60/60))

		os.makedirs(self.saving_path + self.model_dir + self.model_name, exist_ok=True)
		os.makedirs(self.saving_path + self.fig_dir + self.model_name, exist_ok=True)
		os.makedirs(self.saving_path + self.results_dir + self.model_name, exist_ok=True)
		os.makedirs(self.saving_path + self.logfile_dir + self.model_name, exist_ok=True)

	def getKeysInModelName(self):
		keys = {
		'RDIM':'RDIM', 
		'N_used':'N_used', 
		'approx_reservoir_size':'SIZE', 
		'degree':'D', 
		'radius':'RADIUS',
		'sigma_input':'SIGMA',
		'dynamics_length':'DL',
		'noise_level':'NL',
		'iterative_prediction_length':'IPL',
		'regularization':'REG',
		'worker_id':'WID', 
		}
		return keys


	def createModelName(self, params):
		keys = self.getKeysInModelName()
		str_ = "RNN-esn2_" + self.solver
		for key in keys:
			str_ += "-" + keys[key] + "_{:}".format(params[key])
		return str_

	def getSparseWeights(self, sizex, sizey, radius, sparsity, worker_id=1):
		print("WEIGHT INIT")
		W = sparse.random(sizex, sizey, density=sparsity, random_state=worker_id)
		print("EIGENVALUE DECOMPOSITION")
		eigenvalues, eigvectors = splinalg.eigs(W)
		eigenvalues = np.abs(eigenvalues)
		W = (W/np.max(eigenvalues))*radius
		return W

	def augmentHidden(self, h):
		h_aug = h.copy()
		h_aug[::2]=pow(h_aug[::2],2.0)
		return h_aug

	def getAugmentedStateSize(self): 
		return self.reservoir_size


	def train(self):
		self.start_time = time.time()
		dynamics_length = self.dynamics_length
		input_dim = self.input_dim
		N_used = self.N_used

		with open(self.train_data_path, "rb") as file:
			data = pickle.load(file)
			train_input_sequence = data["train_input_sequence"]
			train_target_sequence = data["train_target_sequence"]#!
			print("Adding noise to the training data. {:} per mille ".format(self.noise_level))
			train_input_sequence = addNoise(train_input_sequence, self.noise_level)
			train_target_sequence = addNoise(train_target_sequence, self.noise_level)#!
			N_all, dim = np.shape(train_input_sequence)
			if input_dim > dim: raise ValueError("Requested input dimension is wrong.")
			train_input_sequence = train_input_sequence[:N_used, :input_dim]
			train_target_sequence = train_target_sequence[:N_used]#!
			dt = data["dt"]
			del data
		print("##Using {:}/{:} dimensions and {:}/{:} samples ##".format(input_dim, dim, N_used, N_all))
		if N_used > N_all: raise ValueError("Not enough samples in the training data.")
		print("SCALING")
		
		train_input_sequence, train_target_sequence = self.scaler.scaleData2(train_input_sequence, train_target_sequence)#!

		N, input_dim = np.shape(train_input_sequence)

		print("Initializing the reservoir weights...")
		nodes_per_input = int(np.ceil(self.approx_reservoir_size/input_dim))#*
		self.reservoir_size = int(input_dim*nodes_per_input)#*
		self.sparsity = self.degree/self.reservoir_size;
		print("NETWORK SPARSITY: {:}".format(self.sparsity))
		print("Computing sparse hidden to hidden weight matrix...")
		W_h = self.getSparseWeights(self.reservoir_size, self.reservoir_size, self.radius, self.sparsity, self.worker_id)
		
		W_c = W_h.toarray()
		W_c = np.maximum(W_c, W_c.T)
		degrees = np.zeros(self.reservoir_size)
		for i in range(W_c.shape[0]):
			W_c[i, i] = 0
			neuron_con = W_c[i]
			adjacent_neurons = []
			for j in range(neuron_con.size):
				if neuron_con[j] > 0:
					adjacent_neurons.append(j)
			degrees[i] = len(adjacent_neurons)
		
		def graph_cliques(start_node, set_len, clique_size, node_number, graph, degrees):
			store = np.zeros(node_number, dtype=int)
			cliques = []

			def is_clique(n):
				for i in range(1, n):
					for j in range(i+1, n):  
						if (graph[store[i]][store[j]] == 0):
							return False
				return True
			
			def find_cliques(start_node, set_len, clique_size):
				for j in range(start_node, node_number - (clique_size-set_len)):
					if degrees[j] >= clique_size-1:
						store[set_len] = j
						if is_clique(set_len+1):
							if set_len < clique_size:
								find_cliques(j, set_len + 1, clique_size)
							else:
								clique = [store[i] for i in range(1, set_len+1)]
								cliques.append(clique)
			
			find_cliques(start_node, set_len, clique_size)
			return cliques
		
		clqs = graph_cliques(0, 1, 4, node_number=self.reservoir_size, graph=W_c, degrees=degrees)
		pw_con = graph_cliques(0, 1, 2, node_number=self.reservoir_size, graph=W_c, degrees=degrees)
		print(f"4-Clique number = {len(clqs)}")
		print(f"Pairwise connections = {len(pw_con)}")
		rng = np.random.default_rng(self.worker_id+1)
		clique_con_strengths = rng.random(len(clqs))
		W_c = np.zeros((self.reservoir_size, self.reservoir_size))
		num_i = 0
		for clique in clqs:
			for i in clique:
				for j in clique:
					if i != j:
						W_c[i, j] += (1/3)*clique_con_strengths[num_i]
			num_i += 1
		W_h = W_h.toarray()
		for i in range(W_h.shape[0]):
			W_h[i, i] = 0
		W_h = sparse.coo_array(W_h)
		W_c = minmax_scale(W_c.flatten())
		W_c = W_c.reshape((self.reservoir_size, self.reservoir_size))
		eigenvalues_c, eigvectors_c = splinalg.eigs(W_c)
		eigenvalues_c = np.abs(eigenvalues_c)
		W_c = (W_c/np.max(eigenvalues_c))*self.radius
		W_c *= 0
		

		print("Initializing the input weights...")
		W_in = np.zeros((self.reservoir_size, input_dim))
		q = int(self.reservoir_size/input_dim)
		for i in range(0, input_dim):
			W_in[i*q:(i+1)*q,i] = self.sigma_input * (-1 + 2*np.random.rand(q))

		tl = N - dynamics_length

		print("TRAINING: Dynamics prerun...")
		h = np.zeros((self.reservoir_size, 1))
		for t in range(dynamics_length):
			if self.display_output == True:
				print("TRAINING - Dynamics prerun: T {:}/{:}, {:2.3f}%".format(t, dynamics_length, t/dynamics_length*100), end="\r")
			i = np.reshape(train_input_sequence[t], (-1,1))
			h = np.tanh(self.alpha_1*(W_h @ h) + W_in @ i + self.alpha_2*(W_c @ h))

		print("\n")


		if self.solver == "pinv":
			NORMEVERY = 10
			HTH = np.zeros((self.getAugmentedStateSize(), self.getAugmentedStateSize()))
			YTH = np.zeros((input_dim, self.getAugmentedStateSize()))
		H = []
		Y = []

		print("TRAINING: Teacher forcing...")
		
		for t in range(tl - 1):
			if self.display_output == True:
				print("TRAINING - Teacher forcing: T {:}/{:}, {:2.3f}%".format(t, tl, t/tl*100), end="\r")
			i = np.reshape(train_input_sequence[t+dynamics_length], (-1,1))
			h = np.tanh(self.alpha_1*(W_h @ h) + W_in @ i + self.alpha_2*(W_c @ h))
			h_aug = self.augmentHidden(h)
			H.append(h_aug[:,0])
			target = np.reshape(train_target_sequence[t+dynamics_length+1], (-1,1))#!
			Y.append(target[:,0])
			if self.solver == "pinv" and (t % NORMEVERY == 0):
				H = np.array(H)
				Y = np.array(Y)
				HTH += H.T @ H
				YTH += Y.T @ H
				H = []
				Y = []

		if self.solver == "pinv" and (len(H) != 0):
			H = np.array(H)
			Y = np.array(Y)
			HTH+=H.T @ H
			YTH+=Y.T @ H
			print("TEACHER FORCING ENDED.")
			print(np.shape(H))
			print(np.shape(Y))
			print(np.shape(HTH))
			print(np.shape(YTH))
		else:
			print("TEACHER FORCING ENDED.")
			print(np.shape(H))
			print(np.shape(Y))

		
		print("\nSOLVER used to find W_out: {:}. \n\n".format(self.solver))

		if self.solver == "pinv":
			"""
			Learns mapping H -> Y with Penrose Pseudo-Inverse
			"""
			I = np.identity(np.shape(HTH)[1])	
			pinv_ = scipypinv2(HTH + self.regularization*I)
			W_out = YTH @ pinv_

		elif self.solver in ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag"]:
			"""
			Learns mapping H -> Y with Ridge Regression
			"""
			ridge = Ridge(alpha=self.regularization, fit_intercept=False, normalize=False, copy_X=True, solver=self.solver)
			ridge.fit(H, Y) 
			W_out = ridge.coef_

		else:
			raise ValueError("Undefined solver.")

		print("FINALISING WEIGHTS...")
		self.W_in = W_in
		self.W_h = W_h
		self.W_out = W_out
		self.W_c = W_c
		
		print("COMPUTING PARAMETERS...")
		self.n_trainable_parameters = np.size(self.W_out)
		self.n_model_parameters = np.size(self.W_in) + np.size(self.W_h) + np.size(self.W_out)
		print("Number of trainable parameters: {}".format(self.n_trainable_parameters))
		print("Total number of parameters: {}".format(self.n_model_parameters))
		print("SAVING MODEL...")
		self.saveModel()

	def isWallTimeLimit(self):
		training_time = time.time() - self.start_time
		if training_time > self.reference_train_time:
			print("## Maximum train time reached. ##")
			return True
		else:
			return False

	def predictSequence(self, input_sequence, target_sequence):#!
		W_h = self.W_h
		W_out = self.W_out
		W_in = self.W_in
		W_c = self.W_c
		dynamics_length = self.dynamics_length
		iterative_prediction_length = self.iterative_prediction_length

		self.reservoir_size, _ = np.shape(W_h)
		N = np.shape(input_sequence)[0]
		
		if N != iterative_prediction_length + dynamics_length: raise ValueError("Error! N != iterative_prediction_length + dynamics_length")


		prediction_warm_up = []
		h = np.zeros((self.reservoir_size, 1))
		for t in range(dynamics_length):
			if self.display_output == True:
				print("PREDICTION - Dynamics pre-run: T {:}/{:}, {:2.3f}%".format(t, dynamics_length, t/dynamics_length*100), end="\r")
			i = np.reshape(input_sequence[t], (-1,1))
			h = np.tanh(self.alpha_1*(W_h @ h) + W_in @ i + self.alpha_2*(W_c @ h))
			out = W_out @ self.augmentHidden(h)
			prediction_warm_up.append(out)

		print("\n")

		target = target_sequence[dynamics_length:]#!
		prediction = []
		for t in range(iterative_prediction_length):
			if self.display_output == True:
				print("PREDICTION: T {:}/{:}, {:2.3f}%".format(t, iterative_prediction_length, t/iterative_prediction_length*100), end="\r")
			out = W_out @ self.augmentHidden(h)
			prediction.append(out)
			i = np.vstack((out, input_sequence[t+dynamics_length, 2:].reshape((-1, 1))))#?
			h = np.tanh(self.alpha_1*(W_h @ h) + W_in @ i + self.alpha_2*(W_c @ h))
		print("\n")

		prediction = np.array(prediction)[:,:,0]
		prediction_warm_up = np.array(prediction_warm_up)[:,:,0]

		target_augment = target_sequence#!
		prediction_augment = np.concatenate((prediction_warm_up, prediction), axis=0)

		return prediction, target, prediction_augment, target_augment

	def predictSequenceMemoryCapacity(self, input_sequence, target_sequence):
		W_h = self.W_h
		W_out = self.W_out
		W_in = self.W_in
		W_c = self.W_c
		dynamics_length = self.dynamics_length
		iterative_prediction_length = self.iterative_prediction_length

		self.reservoir_size, _ = np.shape(W_h)
		N = np.shape(input_sequence)[0]
		
		if N != iterative_prediction_length + dynamics_length: raise ValueError("Error! N != iterative_prediction_length + dynamics_length")

		h = np.zeros((self.reservoir_size, 1))
		for t in range(dynamics_length):
			if self.display_output == True:
				print("PREDICTION - Dynamics pre-run: T {:}/{:}, {:2.3f}%".format(t, dynamics_length, t/dynamics_length*100), end="\r")
			i = np.reshape(input_sequence[t], (-1,1))
			h = np.tanh(self.alpha_1*(W_h @ h) + W_in @ i + self.alpha_2*(W_c @ h))
		print("\n")

		target = target_sequence
		prediction = []
		signal = []
		for t in range(dynamics_length, dynamics_length+iterative_prediction_length):
			if self.display_output == True:
				print("PREDICTION: T {:}/{:}, {:2.3f}%".format(t, iterative_prediction_length, t/iterative_prediction_length*100), end="\r")
			signal.append(i)
			out = W_out @ self.augmentHidden(h)
			prediction.append(out)
			i = np.reshape(input_sequence[t], (-1,1))
			h = np.tanh(self.alpha_1*(W_h @ h) + W_in @ i + self.alpha_2*(W_c @ h))
		print("\n")
		prediction = np.array(prediction)[:,:,0]
		target = np.array(target)
		signal = np.array(signal)
		return prediction, target, signal

	def testing(self):
		if self.loadModel()==0:
			self.testingOnTrainingSet()
			self.testingOnTestingSet()
			self.saveResults()
		return 0

	def testingOnTrainingSet(self):
		num_test_ICS = self.num_test_ICS
		with open(self.test_data_path, "rb") as file:
			data = pickle.load(file)
			testing_ic_indexes = data["testing_ic_indexes"]
			dt = data["dt"]
			del data

		with open(self.train_data_path, "rb") as file:
			data = pickle.load(file)
			train_input_sequence = data["train_input_sequence"][:, :self.input_dim]#!
			train_target_sequence = data["train_target_sequence"]#!
			del data
			
		rmnse_avg, num_accurate_pred_005_avg, num_accurate_pred_050_avg, error_freq, predictions_all, truths_all, freq_pred, freq_true, sp_true, sp_pred = self.predictIndexes(train_input_sequence, testing_ic_indexes, dt, "TRAIN", train_target_sequence)#!
		
		for var_name in getNamesInterestingVars():
			exec("self.{:s}_TRAIN = {:s}".format(var_name, var_name))
		return 0

	def testingOnTestingSet(self):
		num_test_ICS = self.num_test_ICS
		with open(self.test_data_path, "rb") as file:
			data = pickle.load(file)
			testing_ic_indexes = data["testing_ic_indexes"]
			test_input_sequence = data["test_input_sequence"][:, :self.input_dim]
			test_target_sequence = data["test_target_sequence"]#!
			dt = data["dt"]
			del data
			
		rmnse_avg, num_accurate_pred_005_avg, num_accurate_pred_050_avg, error_freq, predictions_all, truths_all, freq_pred, freq_true, sp_true, sp_pred = self.predictIndexes(test_input_sequence, testing_ic_indexes, dt, "TEST", test_target_sequence)#!
		
		for var_name in getNamesInterestingVars():
			exec("self.{:s}_TEST = {:s}".format(var_name, var_name))
		return 0


	def predictIndexes(self, input_sequence, ic_indexes, dt, set_name, target_sequence): #!target_sequence
		num_test_ICS = self.num_test_ICS
		input_sequence, target_sequence = self.scaler.scaleData2(input_sequence, target_sequence, reuse=1)#!
		predictions_all = []
		truths_all = []
		rmse_all = []
		rmnse_all = []
		num_accurate_pred_005_all = []
		num_accurate_pred_050_all = []
		for ic_num in range(num_test_ICS):
			if self.display_output == True:
				print("IC {:}/{:}, {:2.3f}%".format(ic_num, num_test_ICS, ic_num/num_test_ICS*100))
			ic_idx = ic_indexes[ic_num]
			input_sequence_ic = input_sequence[ic_idx-self.dynamics_length:ic_idx+self.iterative_prediction_length]
			target_sequence_ic = target_sequence[ic_idx-self.dynamics_length:ic_idx+self.iterative_prediction_length]#!
			prediction, target, prediction_augment, target_augment = self.predictSequence(input_sequence_ic, target_sequence_ic)#!
			prediction, target = self.scaler.descaleData2(prediction, target)#!
			rmse, rmnse, num_accurate_pred_005, num_accurate_pred_050, abserror = computeErrors(target, prediction, self.scaler.target_data_std)
			predictions_all.append(prediction)
			truths_all.append(target)
			rmse_all.append(rmse)
			rmnse_all.append(rmnse)
			num_accurate_pred_005_all.append(num_accurate_pred_005)
			num_accurate_pred_050_all.append(num_accurate_pred_050)

		predictions_all = np.array(predictions_all)
		truths_all = np.array(truths_all)
		rmse_all = np.array(rmse_all)
		rmnse_all = np.array(rmnse_all)
		num_accurate_pred_005_all = np.array(num_accurate_pred_005_all)
		num_accurate_pred_050_all = np.array(num_accurate_pred_050_all)

		print("TRAJECTORIES SHAPES:")
		print(np.shape(truths_all))
		print(np.shape(predictions_all))
		rmnse_avg = np.mean(rmnse_all)
		print("AVERAGE RMNSE ERROR: {:}".format(rmnse_avg))
		num_accurate_pred_005_avg = np.mean(num_accurate_pred_005_all)
		print("AVG NUMBER OF ACCURATE 0.05 PREDICTIONS: {:}".format(num_accurate_pred_005_avg))
		num_accurate_pred_050_avg = np.mean(num_accurate_pred_050_all)
		print("AVG NUMBER OF ACCURATE 0.5 PREDICTIONS: {:}".format(num_accurate_pred_050_avg))
		freq_pred, freq_true, sp_true, sp_pred, error_freq = computeFrequencyError(predictions_all, truths_all, dt)
		print("FREQUENCY ERROR: {:}".format(error_freq))

		return rmnse_avg, num_accurate_pred_005_avg, num_accurate_pred_050_avg, error_freq, predictions_all, truths_all, freq_pred, freq_true, sp_true, sp_pred

	def saveResults(self):

		if self.write_to_log == 1:
			logfile_test = self.saving_path + self.logfile_dir + self.model_name  + "/test.txt"
			writeToTestLogFile(logfile_test, self)
			
		data = {}
		for var_name in getNamesInterestingVars():
			exec("data['{:s}_TEST'] = self.{:s}_TEST".format(var_name, var_name))
			exec("data['{:s}_TRAIN'] = self.{:s}_TRAIN".format(var_name, var_name))
		data["model_name"] = self.model_name
		data["num_test_ICS"] = self.num_test_ICS
		data_path = self.saving_path + self.results_dir + self.model_name + "/results.pickle"
		with open(data_path, "wb") as file:
			pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)
			del data
		return 0

	def loadModel(self):
		data_path = self.saving_path + self.model_dir + self.model_name + "/data.pickle"
		try:
			with open(data_path, "rb") as file:
				data = pickle.load(file)
				self.W_out = data["W_out"]
				self.W_in = data["W_in"]
				self.W_h = data["W_h"]
				self.W_c = data["W_c"]
				self.scaler = data["scaler"]
				del data
			return 0
		except:
			print("MODEL {:s} NOT FOUND.".format(data_path))
			return 1

	def saveModel(self):
		print("Recording time...")
		self.total_training_time = time.time() - self.start_time
		print("Total training time is {:}".format(self.total_training_time))

		print("MEMORY TRACKING IN MB...")
		process = psutil.Process(os.getpid())
		memory = process.memory_info().rss/1024/1024
		self.memory = memory
		print("Script used {:} MB".format(self.memory))
		print("SAVING MODEL...")

		if self.write_to_log == 1:
			logfile_train = self.saving_path + self.logfile_dir + self.model_name  + "/train.txt"
			writeToTrainLogFile(logfile_train, self)

		data = {
		"memory":self.memory,
		"n_trainable_parameters":self.n_trainable_parameters,
		"n_model_parameters":self.n_model_parameters,
		"total_training_time":self.total_training_time,
		"W_out":self.W_out,
		"W_in":self.W_in,
		"W_h":self.W_h,
		"W_c":self.W_c,
		"scaler":self.scaler,
		}
		data_path = self.saving_path + self.model_dir + self.model_name + "/data.pickle"
		with open(data_path, "wb") as file:
			pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)
			del data
		return 0

