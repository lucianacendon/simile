import os
import pickle

def check_train_input(config):
	print ('Checking training input...')

	# Check that all necessary parameters have been defined
	# [DATA]
	try:
		test = config['DATA']
	except:
		print('Header [\'DATA\'] not defined on config file')
		print('Please check the config file available on the repo for reference.')
		exit()

	try:
		list_vars = [config['DATA']['n_features'], config['DATA']['n_target'], config['DATA']['target_labels'], 
					 config['DATA']['train_file'], config['DATA']['model_dir'], config['DATA']['model_label']]
	except KeyError as e:
		print ('Error: parameter', e, 'not defined under [DATA]')
		print ('Please define it on the config file. You may check the example file available on the repo for reference. ')
		exit()

	# Optional variable: if not defined, the initial values will be taken from the training data. 
	try:
		init_val = config['DATA']['init_value'] 
	except:
		config['DATA']['init_value'] = "from_data"

	# [SIMILE]
	try:
		test = config['SIMILE']
	except:
		print('Header [\'SIMILE\'] not defined on config file')
		print('Please check the config file available on the repo for reference.')
		exit()

	try:
		list_vars = [config['SIMILE']['tao'], config['SIMILE']['sigma'], config['SIMILE']['lambd_smooth'], 
		     		config['SIMILE']['n_it'], config['SIMILE']['autoreg_type'], config['SIMILE']['policy_type']]
	except KeyError as e:
		print ('Error: parameter', e, 'not defined under [SIMILE]')
		print ('Please define it on the config file. You may check the example file available on the repo for reference. ')
		exit()

	# [OPERATION_OPTIONS]
	try:
		test = config['OPERATION_OPTIONS']
	except:
		print('Header [\'OPERATION_OPTIONS\'] not defined on config file')
		print('Please check the config file available on the repo for reference.')
		exit()

	try:
		list_vars = [config['OPERATION_OPTIONS']['no_action_first_policy'], config['OPERATION_OPTIONS']['look_future'], 
					config['OPERATION_OPTIONS']['only_env_feat'], config['OPERATION_OPTIONS']['normalize_input'], 
					config['OPERATION_OPTIONS']['normalize_output'], config['OPERATION_OPTIONS']['plot_train_results'],
					config['OPERATION_OPTIONS']['plot_valid_results']]
	except KeyError as e:
		print ('Error: parameter', e, 'not defined under [OPERATION_OPTIONS]')
		print ('Please define it on the config file. You may check the example file available on the repo for reference. ')
		exit()


	# Checking that conditional parameters have been defined
	if (config['SIMILE']['autoreg_type'] == "linear"):
		try:
			test = config['LIN_AUTOREG']
		except:
			print('Header [\'LIN_AUTOREG\'] not defined on config file. This is required for "linear" autoregressor.')
			print('Please check the config file available on the repo for reference.')
			exit()

		try:
			var = config['LIN_AUTOREG']['regularization_autoreg']
		except KeyError:
			print ('Error: regularization of linear autoregressor needs to be defined under [LIN_AUTOREG]')
			print ('Please define it on the config file. You may check the example file available on the repo for reference. ')
			exit()

	if (config['SIMILE']['autoreg_type'] == "geometric_velocity"):

		try:
			test = config['GEOMVEL_AUTOREG']
		except:
			print('Header [\'GEOMVEL_AUTOREG\'] not defined on config file. This is required for "geometric_velocity" autoregressor.')
			print('Please check the config file available on the repo for reference.')
			exit()

		try:
			var = config['GEOMVEL_AUTOREG']['gamma']
		except KeyError:
			print ('Error: gamma for geometric velocity autoregressor needs to be defined under [GEOMVEL_AUTOREG]')
			print ('Please define it on the config file. You may check the example file available on the repo for reference. ')
			exit()

	if (config['SIMILE']['policy_type'] == "xgboost"):
		try:
			test = config['XGBOOST']
		except:
			print('Header [\'XGBOOST\'] not defined on config file. This is required for "xgboost" policy_type.')
			print('Please check the config file available on the repo for reference.')
			exit()

		try:
			list_vars = [config['XGBOOST']['n_estimators'], config['XGBOOST']['regularization'],
						 config['XGBOOST']['learning_rate']]
		except KeyError as e:
			print ('Error: parameter', e, 'not defined under [XGBOOST]')
			print ('Please define it on the config file. You may check the example file available on the repo for reference. ')
			exit()

	if (config['SIMILE']['policy_type'] == "neuralnet"):
		try:
			test = config['NN']
		except:
			print('Header [\'NN\'] not defined on config file. This is required for "neuralnet" policy_type.')
			print('Please check the config file available on the repo for reference.')
			exit()

		try:
			list_vars = [config['NN']['hidsize_1'], config['NN']['hidsize_2'], config['NN']['n_epoch'],
						 config['NN']['save_freq'], config['NN']['batch_size'], config['NN']['learning_rate'],
						 config['NN']['regularization']]
		except KeyError as e:
			print ('Error: parameter', e, 'not defined under [NN]')
			print ('Please define it on the config file. You may check the example file available on the repo for reference. ')
			exit()

	# Checking that valid options have been selected
	# Policy_Type
	if (config['SIMILE']['policy_type'] != "xgboost") and (config['SIMILE']['policy_type'] != "neuralnet"):
		print ('Error: Invalid policy type', config['SIMILE']['policy_type'])
		print ('Policy_type must be either xgboost or neuralnet')
		exit()

	if ((config['SIMILE']['autoreg_type'] != "linear") and (config['SIMILE']['autoreg_type'] != "average") and 
		(config['SIMILE']['autoreg_type'] != "constant") and (config['SIMILE']['autoreg_type'] != "geometric_velocity")):
		print ('Error: Invalid autoregressor type', config['SIMILE']['autoreg_type'])
		print ('autoreg_type must be either linear, average, constant or geometric_velocity')
		exit()

	# Checking some parameters provided
	if (config['DATA']['init_value'] != "from_data"):
		init_pred = [float(i) for i in config['DATA']['init_value'].split(',')]
		try:
			assert(int(config['DATA']['n_target']) == len(init_pred))
		except:
			print ('Error: The number of target variables (' + str(config['DATA']['n_target']) + ') does not match the number of initial values defined in init_value (' + str(len(init_pred)) + ')')
			print ('Please check the config file.')
			exit()

	target_labels = [label for label in config['DATA']['target_labels'].split(',')]
	try:
		assert(int(config['DATA']['n_target']) == len(target_labels))
	except:
		print ('Error: The number of target variables (' + str(config['DATA']['n_target']) + ') does not match the number of target labels defined in target_labels (' + str(len(target_labels)) + ')')
		print ('Please check the config file.')
		exit()

	if (int(config['SIMILE']['tao']) < 2):
		print ('Error: tao should be an integer greater than 2.')
		print ('Please check the config file.')
		exit()

	if (float(config['SIMILE']['sigma']) > 1 or float(config['SIMILE']['sigma']) < 0):
		print ('Error: sigma should be a value between [0, 1]')
		print ('Please check the config file.')
		exit()

	# Making sure model directory exists
	if not os.path.exists(config['DATA']['model_dir']):
		os.makedirs(config['DATA']['model_dir'])

	# Creating directory to store training information
	if not os.path.exists(config['DATA']['model_dir']+'Log/'):
		os.makedirs(config['DATA']['model_dir']+'Log/')

	return

def check_test_input(config):
	print ('Checking test input...')

	# Check that all necessary parameters have been defined
	# [DATA]
	try:
		test = config['DATA']
	except:
		print('Header [\'DATA\'] not defined on config file')
		print('Please check the config file available on the repo for reference.')
		exit()

	try:
		list_vars = [config['DATA']['test_file'], config['DATA']['with_gt'], config['DATA']['model_dir'], 
					 config['DATA']['model_label'], config['DATA']['output_dir'], config['DATA']['policy_load']]
	except KeyError as e:
		print ('Error: parameter', e, 'not defined under [DATA]')
		print ('Please define it on the config file. You may check the example file available on the repo for reference. ')
		exit()

	# Optional variable: if not defined, the initial values will be taken from the ground truth data
	try:
		init_val = config['DATA']['init_value'] 
	except:
		config['DATA']['init_value'] = "from_data"


	# Checking model directory
	if not os.path.exists(config['DATA']['model_dir']):
		print ('Error: ' + str(config['DATA']['model_dir']) + ' does not seem to be a valid directory')
		print ('Please check the path to the trained model defined on the config file (model_dir)')
		exit()

	# Creating directory to save plots
	if not os.path.exists(config['DATA']['output_dir']):
		os.makedirs(config['DATA']['output_dir'])

	try:
		training_param = pickle.load(open(config['DATA']['model_dir'] + config['DATA']['model_label'] + '_train_param.p', 'rb'))
	except FileNotFoundError:
		print ('Error: File "' + str(config['DATA']['model_label']) + '_train_param.p" containing parameters used during training is not present in the model directory ' + str(config['DATA']['model_dir']))
		print ('Please check the config file')
		exit()

	# Checking some parameters provided
	if (config['DATA']['init_value'] != "from_data"):
		init_pred = [float(i) for i in config['DATA']['init_value'].split(',')]
		try:
			assert(int(training_param['n_targets']) == len(init_pred))
		except:
			print ('Error: The number of target parameters (' + str(training_param['n_targets']) + ') used during training does not match the number of initial values defined in init_value (' + str(len(init_pred)) + ')')
			print ('Please check the config file.')
			exit()

	# Checking that valid options have been selected
	if (config['DATA']['policy_load'] != "best_policy"): 
		try:
			int(config['DATA']['policy_load'])
		except ValueError:
			print ('Error: policy_load should be either "best_policy" or an integer corresponding to the policy number to be loaded')
			print ('Please check the config file')
			exit()