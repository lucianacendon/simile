import graph_utils
import configparser
import utils
import numpy as np
import glob

from policy import *
from data_manipulator import SimileDataManipulator
from check_input import *

class SIMILE_PREDICT ():

	def __init__ (self, configfile):

		config = configparser.ConfigParser()
		config.read(configfile)

		# Check that the input provided is compatible with the Simile class
		check_test_input(config)
		print ('Data Checking: complete.')
		
		# Loading parameters used during training
		training_param = pickle.load(open(config['DATA']['model_dir'] + config['DATA']['model_label'] + '_train_param.p', 'rb'))

		## Reading and Formating Parameters ##
		self.data_param, self.simile_param, self.model_param, self.op_param = utils.format_test_parameters (config, training_param)
		print ('Parameter reading: complete.')

		## Loading data
		self.data_manip = SimileDataManipulator(self.data_param, self.simile_param, self.op_param)
		self.data, self.data_param, self.op_param  = self.data_manip.load_test_episodes(config, training_param)
		
		# Normalize if needed
		if (self.op_param['normalize_input'] == 'True') or (self.op_param['normalize_output'] == 'True'):
			self.data, self.data_param = self.data_manip.normalize_test(self.data)
		print ('Data Loading: complete.')

		# Initialize Graphing Utility
		self.graph = graph_utils.SimileGraphUtils(self.data_param, self.op_param)
		if ('Y_test_episodes' in self.data):
			self.graph.A_gt = np.concatenate(self.data['Y_test_episodes'], axis=0)
			self.graph.pol_load = self.simile_param['policy_load']
		else:
			self.graph.A_gt = None
			self.graph.pol_load = self.simile_param['policy_load']

		## Initializing Data Structures ##
		self.initialize_datastructures()

		## Load Trained Models ##
		self.load_policy()


	def initialize_datastructures (self):
		if (self.op_param['look_future'] == 'True'):
			self.X_test = self.data_manip.build_input_structure_with_lookfuture (self.data['X_test_episodes'])
		else:
			self.X_test = self.data_manip.build_input_structure (self.data['X_test_episodes'])

		self.state_shape = (self.X_test.shape[0], (self.X_test.shape[1] + self.data_param['n_targets']*self.simile_param['tao']))

		return

	def load_policy (self):
		# Checking expected policy files
		if (self.model_param['policy_type'] == "neuralnet"):
			list_policies = sorted(glob.glob(self.data_param['model_dir'] + self.data_param['model_label'] + "*.ckpt*"))
			n_policies = int(len(list_policies) / 3)  # Each policy has 3 files 
		elif (self.model_param['policy_type'] == "xgboost"):
			list_policies = sorted(glob.glob(self.data_param['model_dir'] + self.data_param['model_label'] + "*policy*.p"))
			n_policies = int(len(list_policies))  

		else:
			print ('Error: Invalid policy type', self.model_param['policy_type'])
			exit()

		if (n_policies == 0):
			print ('No', self.model_param['policy_type'], 'policy with label', self.data_param['model_label'], 
					'was found at the specified directory', self.data_param['model_dir'])
			exit()
		else:
			print ('Found', n_policies, self.model_param['policy_type'], 'policies with label', self.data_param['model_label'],
					'at the specified directory', self.data_param['model_dir'])

		# Loading policy file paths and autoregressor file paths
		if (self.model_param['policy_type'] == "neuralnet"):
			policy_paths = [self.data_param['model_dir'] + self.data_param['model_label'] + '_NN_policy_' + str(i) + '.ckpt' for i in range(0, n_policies)]
			autoregressor_paths = [self.data_param['model_dir'] + self.data_param['model_label'] + '_autoreg' + str(i) + '.p' for i in range(0, n_policies)]	

			session = tf.InteractiveSession()
			self.model_param.update({'session' : session})	

		elif (self.model_param['policy_type'] == "xgboost"):
			policy_paths = [self.data_param['model_dir'] + self.data_param['model_label'] + '_Xboost_policy' + str(i) + '.p' for i in range(0, n_policies)]
			autoregressor_paths = [self.data_param['model_dir'] + self.data_param['model_label'] + '_autoreg' + str(i) + '.p' for i in range(0, n_policies)]

		# Select only the desired policies
		self.pol_n = self.simile_param['policy_load'] + 1  # 0-indexed

		if (self.pol_n > n_policies) or (self.pol_n < 0):
			print ('Error: Invalid policy', self.pol_n, 'chosen to be loaded')
			print ('policy_load should be a number between 0 and', n_policies)
			print ('Please check the config file')
			exit()

		policy_paths = policy_paths[:self.pol_n]
		autoregressor_paths = autoregressor_paths[:self.pol_n]
		
		assert (len(policy_paths) == len(autoregressor_paths))

		# Loading Policies
		if (n_policies > 1):
			betas = pickle.load(open(self.data_param['model_dir'] + self.data_param['model_label'] + '_betas.p', 'rb'))
			betas = betas[:self.pol_n]
			assert (len(betas) == len(policy_paths))

		print ('Loading', len(policy_paths),' policies')
		self.prev_policies = []
		if (self.pol_n > 1):
			# Reading previous policies
			for i, policy in enumerate(policy_paths[:-1]):
				h = pickle.load(open(autoregressor_paths[i], 'rb'))
				self.model_param.update({'it' : i})
				beta = betas[i]

				print ('Loading Policy ', policy, 'Beta', beta)    

				self.model_param.update({'policy_file' : policy})
				self.policy = Policy(h, self.data_param, self.simile_param, self.model_param,  
							 		 self.op_param, self.prev_policies, None)
				self.policy.beta = beta
				self.prev_policies.append(self.policy)

			# Last Trained Policy
			h = pickle.load(open(autoregressor_paths[-1], 'rb'))
			self.model_param.update({'it' : self.model_param['it'] + 1})
			beta = betas[-1]

			print ('Loading Policy ', policy_paths[-1], 'Beta', beta)

			self.model_param.update({'policy_file' : policy_paths[-1]})
			self.policy = Policy(h, self.data_param, self.simile_param, self.model_param,  
							 	 self.op_param, self.prev_policies, None)
			self.policy.beta = beta
		else:
			# Last Trained Policy
			h = pickle.load(open(autoregressor_paths[-1], 'rb'))
			self.model_param.update({'it' : 0})

			print ('Loading Policy ', policy_paths[-1])
			self.model_param.update({'policy_file' : policy_paths[-1]})
			self.policy = Policy(h, self.data_param, self.simile_param, self.model_param, 
							 	 self.op_param, self.prev_policies, None)


	def policy_rollout (self):
		y_hat, y_hat_raw, a_h = self.policy.rollout (self.X_test, self.data_param['n_dp_epsd_test'], 
													 self.data_param['init_pred_test'], ensemble=True)
		rollout = {'y_hat' : y_hat, 'y_hat_raw' : y_hat_raw, 'a_h' : a_h}

		return rollout

	