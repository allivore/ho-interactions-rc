import numpy as np
import pickle
import io
import os


def getNamesInterestingVars():
	var_names = [
		'rmnse_avg',
		'num_accurate_pred_005_avg',
		'num_accurate_pred_050_avg',
		'error_freq',
		'predictions_all',
		'truths_all',
		'freq_pred',
		'freq_true',
		'sp_true',
		'sp_pred',
	]
	return var_names

def writeToTrainLogFile(logfile_train, model):
	with io.open(logfile_train, 'a+') as f:
		f.write("model_name:" + str(model.model_name)
			+ ":memory:" +"{:.2f}".format(model.memory)
			+ ":total_training_time:" + "{:.2f}".format(model.total_training_time) \
			+ ":n_model_parameters:" + str(model.n_model_parameters) \
			+ ":n_trainable_parameters:" + str(model.n_trainable_parameters) \
			+ "\n"
			)
	return 0

def writeToTestLogFile(logfile_test, model):
	with io.open(logfile_test, 'a+') as f:
		f.write("model_name:" + str(model.model_name)
			+ ":num_test_ICS:" + "{:.2f}".format(model.num_test_ICS)
			+ ":num_accurate_pred_005_avg_TEST:" + "{:.2f}".format(model.num_accurate_pred_005_avg_TEST)
			+ ":num_accurate_pred_050_avg_TEST:" + "{:.2f}".format(model.num_accurate_pred_050_avg_TEST) \
			+ ":num_accurate_pred_005_avg_TRAIN:" + "{:.2f}".format(model.num_accurate_pred_005_avg_TRAIN)
			+ ":num_accurate_pred_050_avg_TRAIN:" + "{:.2f}".format(model.num_accurate_pred_050_avg_TRAIN) \
			+ ":error_freq_TRAIN:" + "{:.2f}".format(model.error_freq_TRAIN) \
			+ ":error_freq_TEST:" + "{:.2f}".format(model.error_freq_TEST) \
			+ "\n"
			)
	return 0

def getReferenceTrainingTime(rtt, btt):
    reference_train_time = 60*60*(rtt-btt)
    print("Reference train time {:} seconds / {:} minutes / {:} hours.".format(rtt, rtt/60, rtt/60/60))
    return reference_train_time


def countTrainableParams(layers):
    temp = 0
    for layer in layers:
        temp+= sum(p.numel() for p in layer.parameters() if p.requires_grad)
    return temp

def printTime(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    h = int(h)
    m = int(m)
    s = int(s)
    print("Time passed: {:d}:{:02d}:{:02d}".format(h, m, s))
    return 0

def countParams(layers):
    temp = 0
    for layer in layers:
        temp+= sum(p.numel() for p in layer.parameters())
    return temp

def splitTrajectories(trajectories, params):
    N_train = int(np.floor(trajectories.size(0) * params["train_valid_ratio"]))
    trajectories_train = trajectories[:N_train]
    trajectories_val = trajectories[N_train:]
    return trajectories_train, trajectories_val


def replaceNaN(data):
	data[np.isnan(data)]=float('Inf')
	return data

def computeErrors(target, prediction, std):
	prediction = replaceNaN(prediction)
	abserror = np.mean(np.abs(target-prediction), axis=1)
	serror = np.square(target-prediction)
	mse = np.mean(serror, axis=1)
	rmse = np.sqrt(mse)
	nserror = serror/np.square(std)
	mnse = np.mean(nserror, axis=1)
	rmnse = np.sqrt(mnse)
	num_accurate_pred_005 = getNumberOfAccuratePredictions(rmnse, 0.05)
	num_accurate_pred_050 = getNumberOfAccuratePredictions(rmnse, 0.5)
	return rmse, rmnse, num_accurate_pred_005, num_accurate_pred_050, abserror

def computeFrequencyError(predictions_all, truths_all, dt):
	sp_pred, freq_pred = computeSpectrum(predictions_all, dt)
	sp_true, freq_true = computeSpectrum(truths_all, dt)
	error_freq = np.mean(np.abs(sp_pred - sp_true))
	return freq_pred, freq_true, sp_true, sp_pred, error_freq

def addNoise(data, percent):
	std_data = np.std(data, axis=0)
	std_data = np.reshape(std_data, (1, -1))
	std_data = np.repeat(std_data, np.shape(data)[0], axis=0)
	noise = np.multiply(np.random.randn(*np.shape(data)), percent/1000.0*std_data)
	data += noise
	return data

class scaler(object):
	def __init__(self, tt):
		self.tt = tt
		self.data_min = 0
		self.data_max = 0
		self.data_mean = 0
		self.data_std = 0       

	def scaleData(self, input_sequence, reuse=None):
		if reuse == None:
			self.data_mean = np.mean(input_sequence,0)
			self.data_std = np.std(input_sequence,0)
			self.data_min = np.min(input_sequence,0)
			self.data_max = np.max(input_sequence,0)
		if self.tt == "MinMaxZeroOne":
			input_sequence = np.array((input_sequence-self.data_min)/(self.data_max-self.data_min))
		elif self.tt == "Standard" or self.tt == "standard":
			input_sequence = np.array((input_sequence-self.data_mean)/self.data_std)
		elif self.tt != "no":
			raise ValueError("Scaler not implemented.")
		return input_sequence

	def descaleData(self, input_sequence):
		if self.tt == "MinMaxZeroOne":
			input_sequence = np.array(input_sequence*(self.data_max - self.data_min) + self.data_min)
		elif self.tt == "Standard" or self.tt == "standard":
			input_sequence = np.array(input_sequence*self.data_std.T + self.data_mean)
		elif self.tt != "no":
			raise ValueError("Scaler not implemented.")
		return input_sequence

	def scaleData2(self, input_sequence, target_sequence, reuse=None):
		if reuse == None:
			self.input_data_mean = np.mean(input_sequence,0)
			self.input_data_std = np.std(input_sequence,0)
			self.input_data_min = np.min(input_sequence,0)
			self.input_data_max = np.max(input_sequence,0)
			self.target_data_mean = np.mean(target_sequence,0)
			self.target_data_std = np.std(target_sequence,0)
			self.target_data_min = np.min(target_sequence,0)
			self.target_data_max = np.max(target_sequence,0)
		if self.tt == "MinMaxZeroOne":
			input_sequence = np.array((input_sequence-self.input_data_min)/(self.input_data_max-self.input_data_min))
			target_sequence = np.array((target_sequence-self.target_data_min)/(self.target_data_max-self.target_data_min))
		elif self.tt == "Standard" or self.tt == "standard":
			input_sequence = np.array((input_sequence-self.input_data_mean)/self.input_data_std)
			target_sequence = np.array((target_sequence-self.target_data_mean)/self.target_data_std)
		elif self.tt != "no":
			raise ValueError("Scaler not implemented.")
		return input_sequence, target_sequence

	def descaleData2(self, input_sequence, target_sequence):
		if self.tt == "MinMaxZeroOne":
			input_sequence = np.array(input_sequence*(self.target_data_max - self.target_data_min) + self.target_data_min)
			target_sequence = np.array(target_sequence*(self.target_data_max - self.target_data_min) + self.target_data_min)
		elif self.tt == "Standard" or self.tt == "standard":
			input_sequence = np.array(input_sequence*self.target_data_std.T + self.target_data_mean)
			target_sequence = np.array(target_sequence*self.target_data_std.T + self.target_data_mean)
		elif self.tt != "no":
			raise ValueError("Scaler not implemented.")
		return input_sequence, target_sequence



def computeSpectrum(data_all, dt):
	spectrum_db = []
	for data in data_all:
		data = np.transpose(data)
		for d in data:
			freq, s_dbfs = dbfft(d, 1/dt)
			spectrum_db.append(s_dbfs)
	spectrum_db = np.array(spectrum_db).mean(axis=0)
	return spectrum_db, freq


def dbfft(x, fs):
	N = len(x)
	if N % 2 != 0:
		x = x[:-1]
		N = len(x)
	x = np.reshape(x, (1,N))
	sp = np.fft.rfft(x)
	freq = np.arange((N / 2) + 1) / (float(N) / fs)
	s_mag = np.abs(sp) * 2 / N
	s_dbfs = 20 * np.log10(s_mag)
	s_dbfs = s_dbfs[0]
	return freq, s_dbfs

def getNumberOfAccuratePredictions(nerror, tresh=0.05):
	nerror_bool = nerror < tresh
	n_max = np.shape(nerror)[0]
	n = 0
	while nerror_bool[n] == True:
		n += 1
		if n == n_max: break
	return n


def addWhiteNoise(data, noise_level):
	std_ = np.std(data, axis=0)
	std_ = np.array(std_).flatten(-1)
	data += np.random.randn(*data.shape)*std_*noise_level/1000.0
	return data


def computeNumberOfModelParameters(variables):
	total_parameters = 0
	for variable in variables:
		shape = variable.get_shape()
		variable_parametes = 1
		for dim in shape:
			variable_parametes *= dim.value
		total_parameters += variable_parametes
	return total_parameters

def isZeroOrNone(var):
	return (var==0 or var==None or var == False or var == str(0))

def stackSequenceData(sequence_data, sequence_length, prediction_length, subsample_seq):
	stacked_input_data = []
	stacked_target_data = []
	if(subsample_seq!=1): print("SEQUENTIALL SUBSAMPLING, ONLY USE IT IN STATE-LESS RNNs WITH LARGE DATA-SETS")
	n = getFirstDataDimension(sequence_data)
	for i in range(0, n - sequence_length - prediction_length, subsample_seq):
		sequence = sequence_data[i:(i+sequence_length), :]
		prediction = sequence_data[(i+sequence_length):(i+sequence_length+prediction_length), :]
		stacked_input_data.append(sequence)
		stacked_target_data.append(prediction)
	return stacked_input_data, stacked_target_data


def stackParallelSequenceData(sequence_data, sequence_length, prediction_length, subsample_seq, parallel_group_interaction_length):
	stacked_input_data = []
	stacked_target_data = []
	pgil = parallel_group_interaction_length
	if(subsample_seq!=1): print("SEQUENTIALL SUBSAMPLING, ONLY USE IT IN STATE-LESS RNNs WITH LARGE DATA-SETS")
	n = getFirstDataDimension(sequence_data)
	for i in range(0, n - sequence_length - prediction_length, subsample_seq):
		sequence = sequence_data[i:(i+sequence_length), :]
		prediction = sequence_data[(i+sequence_length):(i+sequence_length+prediction_length), getFirstActiveIndex(pgil):getLastActiveIndex(pgil)]
		stacked_input_data.append(sequence)
		stacked_target_data.append(prediction)
	return stacked_input_data, stacked_target_data


def getFirstActiveIndex(parallel_group_interaction_length):
	if parallel_group_interaction_length > 0:
		return parallel_group_interaction_length
	else:
		return 0

def getLastActiveIndex(parallel_group_interaction_length):
	if parallel_group_interaction_length > 0:
		return -parallel_group_interaction_length
	else:
		return None

def getFirstDataDimension(var):
	if isinstance(var, (list,)):
		dim = len(var)
	elif type(var) == np.ndarray:
		dim = np.shape(var)[0]
	elif  type(var) == np.matrix:
		raise ValueError("Variable is a matrix. NOT ALLOWED!")
	else:
		raise ValueError("Variable not a list or a numpy array. No dimension to compute!")
	return dim

def divideData(data, train_val_ratio):
	n_samples = getFirstDataDimension(data)
	n_train = int(n_samples*train_val_ratio)
	data_train = data[:n_train]
	data_val = data[n_train:]
	return data_train, data_val

def createTrainingDataBatches(input_train, target_train, batch_size):
	n_samples = getFirstDataDimension(input_train)
	input_train_batches = []
	target_train_batches = []
	n_batches = int(n_samples/batch_size)
	for i in range(n_batches):
		input_train_batches.append(input_train[batch_size*i:batch_size*i+batch_size])
		target_train_batches.append(target_train[batch_size*i:batch_size*i+batch_size])
	return input_train_batches, target_train_batches, n_batches

def subsample(data, max_samples):
	n_samples = getFirstDataDimension(data)
	if n_samples>max_samples:
		step = int(np.floor(n_samples/max_samples))
		if step == 1:
			data = data[:max_samples]
		else:
			data = data[::step][:max_samples]
	return data
