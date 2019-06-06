import utils
import graph_utils
import pickle
import numpy as np 
import configparser
import tensorflow as tf

from autoregressor import *
from policy import *
from collections import deque
from data_manipulator import SimileDataManipulator
from check_input import *

class SIMILE_TRAIN ():

	def __init__ (self, configfile):
		config = configparser.ConfigParser()
		config.read(configfile)

		# Check that the input provided is compatible with the Simile class
		check_train_input(config)
		print ('Data Checking: complete.')

		## Reading and Formating Parameters ##
		self.data_param, self.simile_param, self.learning_param, self.model_param, self.op_param = utils.format_train_parameters (config)
		print ('Parameter reading: complete.')

		## Loading data
		self.data_manip = SimileDataManipulator(self.data_param, self.simile_param, self.op_param)
		self.data, self.data_param, self.op_param = self.data_manip.load_train_episodes(config)

		# Normalize if needed
		if (self.op_param['normalize_input'] == 'True') or (self.op_param['normalize_output'] == 'True'):
			self.data, self.data_param = self.data_manip.normalize_train(self.data)
		print ('Data Loading: complete.')

		# Initialize Graphing Utility
		if (self.op_param['plot_train_results'] == 'True') or (self.op_param['plot_valid_results'] == 'True'):
			self.graph = graph_utils.SimileGraphUtils(self.data_param, self.op_param)

		## Initializing Data Structures ##
		self.initialize_datastructures()

		print ('----- DataStructures Initialized -----')
		print ('Time Horizon:', self.simile_param['tao'])
		print ('-- Training --')
		print ('Input Features: ', self.X_train.shape)
		print ('Actions: ', self.A_train_gt.shape)

		if (self.op_param['only_env_feat'] == 'False') and (self.op_param['no_action_first_policy'] == 'False'):
			print ('Actions History: ', self.A_train_hist.shape)
			print ('State [Input Feature + Actions History]: ', self.S_train.shape)
		else:
			print ('State [Input Features]: ', self.S_train.shape)
		
		if (self.op_param['include_validation']):
			print ('-- Validation --')
			print ('Input Features: ', self.X_valid.shape)
			print ('Actions: ', self.A_valid_gt.shape)
			if (self.op_param['only_env_feat'] == 'False') and (self.op_param['no_action_first_policy'] == 'False'):
				print ('Actions History: ', self.A_valid_hist.shape)
				print ('State [Input Feature + Actions History]: ', self.S_valid.shape)
			else:
				print ('State [Input Features]: ', self.S_valid.shape)

		print ('-------------------------------------')


	def initialize_datastructures (self):

		# Building Input X with history information
		if (self.op_param['look_future'] == 'True'):
			self.X_train = self.data_manip.build_input_structure_with_lookfuture (self.data['X_train_episodes'])
			if (self.op_param['include_validation']):
				self.X_valid = self.data_manip.build_input_structure_with_lookfuture (self.data['X_valid_episodes'])
		else:
			self.X_train = self.data_manip.build_input_structure (self.data['X_train_episodes'])
			if (self.op_param['include_validation']):
				self.X_valid = self.data_manip.build_input_structure (self.data['X_valid_episodes'])

		# Building initial action structure with history information
		self.A_train_hist, self.A_train_gt = self.data_manip.build_init_action_structure (self.data['Y_train_episodes'])
		assert(self.A_train_hist.shape[1] == (self.simile_param['tao']-1)*self.data_param['n_targets'])
		assert(self.A_train_gt.shape[1] == self.data_param['n_targets'])

		if (self.op_param['include_validation']):
			self.A_valid_hist, self.A_valid_gt = self.data_manip.build_init_action_structure (self.data['Y_valid_episodes'])
			assert(self.A_valid_hist.shape[1] == (self.simile_param['tao']-1)*self.data_param['n_targets'])
			assert(self.A_valid_gt.shape[1] == self.data_param['n_targets'])

		# Building state variables
		if (self.op_param['only_env_feat'] == 'True') or (self.op_param['no_action_first_policy'] == 'True'):
			self.S_train = self.X_train
			if (self.op_param['include_validation']):
				self.S_valid = self.X_valid
		else:			
			self.S_train = self.form_states (self.X_train, self.A_train_hist)
			assert (self.S_train.shape[1] == self.X_train.shape[1] + self.A_train_hist.shape[1])
			if (self.op_param['include_validation']):
				self.S_valid = self.form_states (self.X_valid, self.A_valid_hist)
				assert (self.S_valid.shape[1] == self.X_valid.shape[1] + self.A_valid_hist.shape[1])
		
		# Calculating expert derivatives to help choosing best policy
		self.expert_derivatives = self.data_manip.get_expert_derivatives(self.A_train_gt)

		return

	def form_states (self, X, A):
		return np.concatenate((X, A), axis = 1)

	def collect_smooth_feedback (self, A_rollout, A_gt):
		return self.sigma*A_rollout + (1-self.sigma)*A_gt

	def fit_autoregressor (self, action_hist, action_target, it):
		print ('Fitting Autoregressor: iteration', it, '...')

		target_idx = []
		autoreg = []

		all_idx = np.arange(1, action_hist.shape[1] + 1)
		for i in range(1, self.data_param['n_targets'] + 1):
			target_idx.append([j-1 for j in all_idx if (j+(self.data_param['n_targets']-i)) % self.data_param['n_targets'] == 0])

		assert (action_hist[:, target_idx[0]].shape[1] == (self.simile_param['tao']-1))

		for i in range(self.data_param['n_targets']):
			curr_autoreg = Autoregressor() 
			if(self.simile_param['autoreg_type'] == "linear"):
				curr_autoreg.set_linear(self.simile_param['regularization_autoreg'], action_hist[:, target_idx[i]], action_target[:, i])
			elif(self.simile_param['autoreg_type'] == "constant"):
				curr_autoreg.set_constant()
			elif(self.simile_param['autoreg_type'] == "average"):
				curr_autoreg.set_average()
			elif(self.simile_param['autoreg_type'] == "geometric_velocity"):
				curr_autoreg.set_geometric_velocity(self.simile_param['gamma'])
			else:
				print ("Error: Invalid autoregressor type", self.simile_param['autoreg_type'])
				exit()
			autoreg.append(curr_autoreg)

		assert(len(autoreg) == self.data_param['n_targets'])
			
		for i in range(action_target.shape[1]):
			pickle.dump(autoreg, open(self.data_param['output_dir'] + self.data_param['model_label'] + '_autoreg' + str(it) + '.p', 'wb'))
		
		return autoreg

	def policy_rollout (self, it):
		y_hat_train, y_hat_raw_train, a_h_train = self.policy.rollout (self.X_train, self.data_param['n_dp_epsd_train'], 
																	   self.data_param['init_pred_train'], ensemble=True)
		rollout = {'y_hat_train' : y_hat_train, 'y_hat_raw_train' : y_hat_raw_train, 'a_h_train' : a_h_train}	

		if (self.op_param['include_validation']):
			y_hat_valid, y_hat_raw_valid, a_h_valid = self.policy.rollout (self.X_valid, self.data_param['n_dp_epsd_test'], 
																		   self.data_param['init_pred_valid'], ensemble=True)
			rollout.update({'y_hat_valid' : y_hat_valid, 'y_hat_raw_valid' : y_hat_raw_valid, 'a_h_valid' : a_h_valid})

		return rollout

	def get_action_hist (self, A_rollout, n_frames_per_episode):
		""" Builds action structure from policy roll out """

		# Building action history structure to further define state S
		frame_n = 0
		episode_n = 0
		A = []
		for action in A_rollout:
			# Identifying new episode from rollout structure
			if (frame_n == n_frames_per_episode[episode_n]):
				frame_n = 0
				episode_n += 1

			if (frame_n == 0):
				# Initializing new deque for new episode
				recent_actions = deque()
				for t in range(self.simile_param['tao']): 
					recent_actions.append(action)

			# Remove last element (not part of context anymore)	
			recent_actions.popleft()	
			# Add current action to datastructure - this is (at), while the rest of structure is (at-1:t-tao)
			recent_actions.append(action)
			# Flatten deque structure
			action_array = np.hstack(recent_actions)
			# Ignores the most recent one (current action instead of history)
			A.append(action_array[:self.data_param['n_targets']*(self.simile_param['tao']-1)])

			frame_n += 1

		return np.concatenate(A,  axis=0).reshape(A_rollout.shape[0], self.data_param['n_targets']*(self.simile_param['tao']-1))

	def sum_squared_error(self, rollout):
		return np.sum(abs((rollout - self.A_train_gt)**2))

	def roughness_coeff (self, rollout):

		assert (rollout.shape[0] == self.A_train_gt.shape[0])

		rollout_derivatives = np.zeros((rollout.shape), dtype='float64')
		for i in range(1, rollout.shape[0]):
			rollout_derivatives[i] = rollout[i] - rollout[i-1]

		return np.sum(abs(rollout_derivatives - self.expert_derivatives))
		

	def check_best_policy (self, rollout, it):
		curr_sse = self.sum_squared_error (rollout)
		curr_rough = self.roughness_coeff (rollout)
		curr_coeff = (curr_sse + self.simile_param['lambda_loss']*curr_rough)/(1+self.simile_param['lambda_loss'])

		if (it == 0):
			print ('Sum Squared Error w.r.t Expert Intial Policy: ', curr_sse)
			print ('Roughness Coeff. Initial Policy: ', curr_rough)
			print ('Quality Coeff. Initial Policy: ', curr_coeff)
			self.data_param['log_file'].writelines("Sum Squared Error w.r.t Expert Initial Policy: {:f} \n".format(curr_sse))
			self.data_param['log_file'].writelines("Roughness Coeff. of Initial Policy: {:f} \n".format(curr_rough))
			self.data_param['log_file'].writelines("Quality Coeff. Initial Policy: {:f} \n".format(curr_coeff))

			# Saving best policy 
			self.best_policy = {'coeff' : curr_coeff, 'pol_n' : 0}
			training_param = pickle.load(open(self.data_param['output_dir'] + self.data_param['model_label'] + '_train_param.p', 'rb'))
			training_param.update({'best_policy' : self.best_policy['pol_n']})
			pickle.dump(training_param, open(self.data_param['output_dir'] + self.data_param['model_label'] + '_train_param.p', 'wb'))

		else:
			print ('Sum Squared Error w.r.t Expert Current Policy: ', curr_sse)
			print ('Roughness Coeff. Current Policy: ', curr_rough)
			print ('Quality Coeff. Current Policy: ', curr_coeff)
			self.data_param['log_file'].writelines("Sum Squared Error w.r.t Expert Current Policy: {:f} \n".format(curr_sse))
			self.data_param['log_file'].writelines("Roughness Coeff. Current Policy: {:f} \n".format(curr_rough))
			self.data_param['log_file'].writelines("Quality Coeff. Current Policy: {:f} \n".format(curr_coeff))

			if (curr_coeff < self.best_policy['coeff']):
				# Saving best policy 
				print ('New Best Policy: policy', it)
				self.data_param['log_file'].writelines('New Best Policy: policy {:1d} \n'.format(it))
				self.best_policy['coeff'] = curr_coeff
				self.best_policy['pol_n'] = it

				training_param = pickle.load(open(self.data_param['output_dir'] + self.data_param['model_label'] + '_train_param.p', 'rb'))
				training_param.update({'best_policy' : self.best_policy['pol_n']})
				pickle.dump(training_param, open(self.data_param['output_dir'] + self.data_param['model_label'] + '_train_param.p', 'wb'))
		return 

	def train (self):
		# Train Initial autoregressor
		self.h = self.fit_autoregressor(self.A_train_hist, self.A_train_gt, it=0)
		self.model_param.update({'it' : 0})

		if (self.model_param['policy_type'] == 'neuralnet'):
			self.session = tf.InteractiveSession()
			self.model_param.update({'session' : self.session})

		# Train Initial Policy
		self.prev_policies = []
		self.policy = Policy(self.h, self.data_param, self.simile_param, self.model_param,
							 self.op_param, self.prev_policies, self.learning_param)

		States = {'S_train' : self.S_train}
		Actions = {'A_train' : self.A_train_gt, 'A_train_gt' : self.A_train_gt}
		if (self.op_param['include_validation']):
			States.update({'S_valid' : self.S_valid})
			Actions.update({'A_valid' : self.A_valid_gt})
			Actions.update({'A_valid_gt' : self.A_valid_gt})

		self.policy.model.train(States, Actions)

		# Rolling out trained policy
		rollout = self.policy_rollout(0)
		if (self.op_param['plot_train_results'] == 'True') or (self.op_param['plot_valid_results'] == 'True'):
			self.graph.plot_rollout_train(rollout, Actions, it=0)
		self.A_rollout = {'train' : rollout['y_hat_train']}
		if (self.op_param['include_validation']):
			self.A_rollout.update({'valid' : rollout['y_hat_valid']})

		# Initiazing beta
		self.policy.beta = 1
		self.all_betas = [self.policy.beta]
		self.prev_policies.append(self.policy)

		# Initialize best policy
		self.check_best_policy(self.A_rollout['train'] , 0)

		# Train for Multiple Iterations
		for it in range(1, self.simile_param['n_it']+1):
			# Building action history structures from rollout
			self.A_train_hist = self.get_action_hist (self.A_rollout['train'], self.data_param['n_dp_epsd_train']) 
			if (self.op_param['include_validation']):
				self.A_valid_hist = self.get_action_hist (self.A_rollout['valid'], self.data_param['n_dp_epsd_test'])

			# Form new states 
			self.S_train  = self.form_states (self.X_train, self.A_train_hist)
			if (self.op_param['include_validation']):
				self.S_valid = self.form_states (self.X_valid, self.A_valid_hist)

			# Smooth feedback
			self.sigma = self.simile_param['sigma']/(1.6**(it-1))
			print ('Sigma', self.sigma)
			self.A_train = self.collect_smooth_feedback (self.A_rollout['train'], self.A_train_gt)
			if (self.op_param['include_validation']):
				self.A_valid = self.collect_smooth_feedback (self.A_rollout['valid'], self.A_valid_gt) 

			# Fit Autoregressor 
			self.h = self.fit_autoregressor (self.A_train_hist, self.A_train, it)
			self.model_param.update({'it' : it})

			# Training new policy
			self.policy = Policy(self.h, self.data_param, self.simile_param, self.model_param,
							 	 self.op_param, self.prev_policies, self.learning_param)

			# Update States and Action structures
			States['S_train'] = self.S_train
			Actions['A_train'] = self.A_train
			if (self.op_param['include_validation']):
				States['S_valid'] = self.S_valid
				Actions['A_valid'] = self.A_valid
			
			# Update Betas
			self.policy.model.train(States, Actions)
			self.policy.update_beta(self.X_train, self.A_rollout['train'], self.A_train_gt)
			self.prev_policies.append(self.policy)
			self.all_betas.append(self.policy.beta)

			# Rolling out ensemble of policies (included new one, with updated beta)
			rollout = self.policy_rollout(it)
			self.A_rollout['train'] = rollout['y_hat_train']
			if (self.op_param['plot_train_results'] == 'True') or (self.op_param['plot_valid_results'] == 'True'):
				self.graph.plot_rollout_train(rollout, Actions, it)
			if (self.op_param['include_validation']):
				self.A_rollout['valid'] = rollout['y_hat_valid']

			# Updating best policy if needed
			self.check_best_policy(self.A_rollout['train'], it)
			
			# Plot Fitting results
			if (self.op_param['plot_train_results'] == 'True'):
				self.graph.plot_smoothfeedback (self.A_rollout['train'], self.A_train, self.A_train_gt, 'train', it)
			if (self.op_param['include_validation']) and (self.op_param['plot_valid_results'] == 'True'):
				self.graph.plot_smoothfeedback (self.A_rollout['valid'], self.A_valid, self.A_valid_gt, 'valid', it)

			# Saving betas for later model restoring
			pickle.dump(self.all_betas, open(self.data_param['output_dir'] + self.data_param['model_label'] + '_betas.p', 'wb'))
			

		self.data_param['log_file'].close() 
		plt.close('all')

