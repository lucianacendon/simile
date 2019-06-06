import pickle
import numpy as np
import matplotlib.pyplot as plt
import configparser

def normalize_train_data (data):
	n_columns = data.shape[1]
	n_rows = data.shape[0]

	# Initializing structures
	data_norm = np.zeros((data.shape))
	min_col = np.zeros(n_columns)
	max_col = np.zeros(n_columns)

	for j in range(n_columns):
		min_col[j] = np.min(data[:, j])
		max_col[j] = np.max(data[:, j])

	for j in range(n_columns):
		for i in range(n_rows):
			data_norm[i, j] = (data[i, j] - min_col[j]) / (max_col[j] - min_col[j]) - 0.5

	return data_norm, min_col, max_col


def normalize_test_data (data, min_col, max_col):
	n_columns = data.shape[1]
	n_rows = data.shape[0]

	# Initializing structures
	data_norm = np.zeros((data.shape))
		 
	# Normalizing data according to min and max values from the training set
	for j in range(n_columns):
		for i in range(n_rows):
			data_norm[i, j] = (data[i, j] - min_col[j]) / (max_col[j] - min_col[j]) - 0.5

	return data_norm


def denormalize (data, min_Y, max_Y):

	n_columns = data.shape[1]
	n_rows = data.shape[0]

	assert (n_columns == min_Y.shape[0] == max_Y.shape[0])

	data_denorm = np.zeros((data.shape))

	for j in range(n_columns):
		for i in range(n_rows):
			data_denorm[i, j] = (data[i, j] + 0.5)*(max_Y[j] - min_Y[j]) + min_Y[j]

	return data_denorm	


def format_train_parameters (config):
	target_labels = [label for label in config['DATA']['target_labels'].split(',')]
	if (config['DATA']['init_value'] != "from_data"):
		init_pred = [float(i) for i in config['DATA']['init_value'].split(',')]
	else:
		init_pred = "from_data"
		
	log_dir = config['DATA']['model_dir'] + 'Log/'
	log_file = open(log_dir + 'Training_Log_Info.txt', 'w')

	data_param = {'n_features' : int(config['DATA']['n_features']), 
				  'n_targets' : int(config['DATA']['n_target']),
				  'target_labels' : target_labels,
				  'init_pred' : init_pred,
				  'output_dir' : config['DATA']['model_dir'],
				  'model_label' : config['DATA']['model_label'],
				  'log_dir' : log_dir, 
				  'log_file' : log_file }

	op_param = { 'no_action_first_policy' : config['OPERATION_OPTIONS']['no_action_first_policy'],
				 'look_future' : config['OPERATION_OPTIONS']['look_future'],
				 'only_env_feat' : config['OPERATION_OPTIONS']['only_env_feat'],
				 'normalize_input' : config['OPERATION_OPTIONS']['normalize_input'],
				 'normalize_output' : config['OPERATION_OPTIONS']['normalize_output'],
				 'plot_train_results' : config['OPERATION_OPTIONS']['plot_train_results'],
				 'plot_valid_results' : config['OPERATION_OPTIONS']['plot_valid_results'],}

	simile_param = {'tao' : int(config['SIMILE']['tao']), 'sigma' : float(config['SIMILE']['sigma']), 
					'lambda_loss' : float(config['SIMILE']['lambd_smooth']), 'n_it' : int(config['SIMILE']['n_it']), 
					'autoreg_type' : config['SIMILE']['autoreg_type'] }

	if (op_param['only_env_feat'] == 'True'):
		simile_param.update({'n_it' : 0 }) # Stops at Policy 0

	if (config['SIMILE']['autoreg_type'] == "linear"):
		simile_param.update({'regularization_autoreg' : float(config['LIN_AUTOREG']['regularization_autoreg'])})

	if (config['SIMILE']['autoreg_type'] == "geometric_velocity"):
		simile_param.update({'gamma' : config['SIMILE']['autoreg_type'] })

	if (config['SIMILE']['policy_type'] == "neuralnet"):

		learning_param = {	'n_epoches' : int(config['NN']['n_epoch']), 
							'learning_rate' : float(config['NN']['learning_rate']), 
							'batch_size' : int(config['NN']['batch_size']),
							'save_freq' : int(config['NN']['save_freq']) }
		model_param = {	'policy_type' : config['SIMILE']['policy_type'],
						'hidsize_1' : int(config['NN']['hidsize_1']),
						'hidsize_2' : int(config['NN']['hidsize_2']), 
						'regularization' : float(config['NN']['regularization']) }

	elif (config['SIMILE']['policy_type'] == "xgboost"):

		learning_param = {'learning_rate' : float(config['XGBOOST']['learning_rate'])}
		model_param = {'policy_type' : config['SIMILE']['policy_type'],
					   'n_estimators' : int(config['XGBOOST']['n_estimators']),
					   'regularization' : float(config['XGBOOST']['regularization'])}
	else:
		print ('Error: Invalid policy type ', config['SIMILE']['policy_type'])
		print ('Policy Type should be either "neuralnet" or "xgboost"')
		exit()

	save_training_parameters(config, model_param)
	model_param.update({'train' : True})

	return data_param, simile_param, learning_param, model_param, op_param


def format_test_parameters (config, training_param):
	if (config['DATA']['init_value'] != "from_data"):
		init_pred = [float(i) for i in config['DATA']['init_value'].split(',')]
	else:
		init_pred = "from_data"

	data_param = {'n_features' : training_param['n_features'], 
				  'n_targets' : training_param['n_targets'],
				  'target_labels' : training_param['target_labels'],
				  'init_pred' : init_pred,
				  'model_label' : config['DATA']['model_label'],
				  'model_dir' : config['DATA']['model_dir'],
				  'output_dir' : config['DATA']['model_dir'],
				  'log_dir' : config['DATA']['output_dir']}

	simile_param = {'tao' : training_param['tao'], 
					'lambda_loss' : training_param['lambda_loss'], 
					'autoreg_type' : training_param['autoreg_type'] }

	if (config['DATA']['policy_load'] == "best_policy"):
		simile_param.update({'policy_load' : training_param['best_policy']})
	else:
		simile_param.update({'policy_load' : int(config['DATA']['policy_load'])})

	if (training_param['policy_type'] == "neuralnet"):

		model_param = { 'policy_type' : training_param['policy_type'],
						'hidsize_1' : training_param['model_param']['hidsize_1'],
						'hidsize_2' : training_param['model_param']['hidsize_2'],
						'regularization' : training_param['model_param']['regularization'] }

	elif (training_param['policy_type'] == "xgboost"):

		model_param = { 'policy_type' : training_param['policy_type'],
						'n_estimators' : training_param['model_param']['n_estimators'],
						'regularization' : training_param['model_param']['regularization'] }
	else:
		print ("Error: Invalid policy type ", policy_type)
		exit()

	model_param.update({'train' : False})

	op_param = { 'no_action_first_policy' : training_param['no_action_first_policy'],
				 'look_future' : training_param['look_future'],
				 'only_env_feat' : training_param['only_env_feat'],
				 'normalize_input' : training_param['normalize_input'],
				 'normalize_output' : training_param['normalize_output'] }

	return data_param, simile_param, model_param, op_param


def save_training_parameters(config, model_param):
	target_labels = [label for label in config['DATA']['target_labels'].split(',')]

	training_param = { 'tao' : int(config['SIMILE']['tao']),
					'lambda_loss' : float(config['SIMILE']['lambd_smooth']),
					'autoreg_type' : config['SIMILE']['autoreg_type'],
					'policy_type' : config['SIMILE']['policy_type'],
					'model_param' : model_param,
					'n_features' : int(config['DATA']['n_features']), 
					'n_targets' : int(config['DATA']['n_target']),
					'target_labels' : target_labels,
					'no_action_first_policy' : config['OPERATION_OPTIONS']['no_action_first_policy'],
					'look_future' : config['OPERATION_OPTIONS']['look_future'],
					'only_env_feat' : config['OPERATION_OPTIONS']['only_env_feat'],
					'normalize_input' : config['OPERATION_OPTIONS']['normalize_input'],
					'normalize_output' : config['OPERATION_OPTIONS']['normalize_output'] }

	pickle.dump(training_param, open(str(config['DATA']['model_dir']) + config['DATA']['model_label'] + '_train_param.p', 'wb'))

	return