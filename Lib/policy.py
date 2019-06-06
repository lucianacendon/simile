import numpy as np

from parallel_tree_boosting import *
from neuralnet import *
from collections import deque

class Policy():

	def __init__(self, h, data_param, simile_param, model_param, 
				 op_param, prev_policies, learning_param):
		self.h = h
		self.it = model_param['it']

		self.policy_type = model_param['policy_type']
		self.data_param = data_param
		self.simile_param = simile_param
		self.model_param = model_param
		self.op_param = op_param
		self.learning_param = learning_param
		self.prev_policies = prev_policies

		if (self.policy_type == "xgboost"):
			self.model = XGBoost(data_param, model_param, learning_param)
		elif (self.policy_type == "neuralnet"):
			if (self.op_param['only_env_feat'] == 'True') or ((self.op_param['no_action_first_policy'] == 'True') and self.it == 0):
				input_shape = self.data_param['n_features']*self.simile_param['tao']
			else:
				input_shape = (self.data_param['n_features']+(self.data_param['n_targets']))*self.simile_param['tao'] - (self.data_param['n_targets'])
			
			self.model = NeuralNet(input_shape, data_param, model_param, learning_param)
		else:
			print ("Error: Invalid policy_type", self.policy_type)
			exit()


	def rollout (self, X, n_datapoints_episode, init_pred, ensemble):
		if (ensemble):
			print ('Rolling out ensemble policy ...')
		else:
			print ('Rolling out current policy...')

		n_datapoints = X.shape[0]

		# Making sure the number of datapoints look correct before proceeding
		assert (n_datapoints == np.sum(n_datapoints_episode))

		# Initializing structures
		y_hat = np.zeros((n_datapoints, self.data_param['n_targets']))
		y_hat_raw = np.zeros((n_datapoints, self.data_param['n_targets']))
		a_h = np.zeros((n_datapoints, self.data_param['n_targets']))

		dp_n = 0
		episode = -1
		for i in range(0, n_datapoints):
			# This means beginning of new episode
			if (dp_n == n_datapoints_episode[episode]) or (episode == -1):
				dp_n = 0
				episode += 1

				# Re-initialize deque when new episode		
				recent_actions = deque()
				for t in range(self.simile_param['tao']-1):
					recent_actions.append(init_pred[episode])

				y_hat[i] = init_pred[episode]
				y_hat_raw[i] = init_pred[episode]
				a_h[i] = init_pred[episode]
			else:
				state = np.concatenate( (X[i].reshape(1, -1), np.hstack(recent_actions).reshape(1, -1)), axis = 1)

				if (ensemble):
					y_hat[i], y_hat_raw[i], a_h[i] = self.predict_ensemble (state, self.prev_policies) 
				else:
					y_hat[i], y_hat_raw[i], a_h[i] = self.get_smooth_prediction (state)

			recent_actions.popleft()
			recent_actions.append(y_hat[i])

			dp_n += 1

		assert(dp_n == n_datapoints_episode[-1])

		return y_hat, y_hat_raw, a_h


	def get_smooth_prediction (self, state):

		if (self.op_param['only_env_feat'] == 'True') or ((self.op_param['no_action_first_policy'] == 'True') and self.it == 0):
			state_raw = state[:, :self.data_param['n_features']*self.simile_param['tao']]	# Only feature part of the state 
		else:
			state_raw = state[:]

		# Action predicted by model
		y_hat_raw = self.model.get_raw_prediction (state_raw)

		# Action slice of the state
		a_hist = state[:, -(self.simile_param['tao']-1)*self.data_param['n_targets']:]

		# Spliting autoregressor
		target_idx = []
		all_idx = np.arange(1, a_hist.shape[1] + 1)
		for i in range(1, self.data_param['n_targets'] + 1):
			target_idx.append([j-1 for j in all_idx if (j+(self.data_param['n_targets']-i)) % self.data_param['n_targets'] == 0])

		assert (a_hist[:, target_idx[0]].shape[1] == (self.simile_param['tao']-1))

		# Action predicted by the autoregressor
		a_h = np.zeros((a_hist.shape[0], self.data_param['n_targets']))
		for i in range(0, self.data_param['n_targets']):
			a_h[:, i] = self.h[i].predict(a_hist[:, target_idx[i]])

		# Smooth rollout
		y_hat = np.zeros((y_hat_raw.shape))
		y_hat = (y_hat_raw + self.simile_param['lambda_loss']*a_h) / (1 + self.simile_param['lambda_loss'])

		return y_hat, y_hat_raw, a_h


	def update_beta(self, X, A_rollout_prev_policy, A_train_gt):

		# MSE rollout of previous ensemble policy w.r.t ground truth
		error_prev_policy = np.sqrt(np.mean((A_rollout_prev_policy - A_train_gt)**2))

		# MSE rollout of current policy w.r.t ground truth
		A_rollout_curr_policy = self.rollout (X, self.data_param['n_dp_epsd_train'], self.data_param['init_pred_train'], ensemble=False)
		error_policy = np.sqrt(np.mean((A_rollout_curr_policy - A_train_gt)**2))

		self.beta = error_prev_policy / (error_prev_policy + error_policy)

		print ('MSE w.r.t Expert from Prev. Policy: ', error_prev_policy)
		print ('MSE w.r.t Expert from Current Policy: ', error_policy)
		self.data_param['log_file'].writelines("MSE w.r.t Expert from Prev. Policy: : {:f} \n".format(error_prev_policy))
		self.data_param['log_file'].writelines("MSE w.r.t Expert from Current Policy: {:f} \n".format(error_policy))

		# Writing information 
		print ('Policy {:1d} - Beta Weight: {:f}'.format(self.it, self.beta))
		self.data_param['log_file'].writelines('Policy {:1d} - Beta Weight: {:f} \n\n'.format(self.it, self.beta))
		
		return 

	def predict_ensemble (self, S, prev_policies):
		y_hat, y_hat_raw, a_h = self.get_smooth_prediction(S)

		if (prev_policies):
			prev_policy = prev_policies[-1]
			y_hat_prev, y_hat_raw_prev, a_h_prev = prev_policy.predict_ensemble(S, prev_policies[:-1])
			return self.beta*y_hat + (1-self.beta)*y_hat_prev , y_hat_raw, a_h

		return y_hat, y_hat_raw, a_h

	def error_ensemble (self, S, A, prev_policies):
		y_hat, y_hat_raw, a_h = self.predict_ensemble(S, prev_policies)
		return np.sqrt(np.mean((y_hat-A)**2))

	def error_smooth_current_model (self, S, A):
		y, y_raw, a_h = self.get_smooth_prediction(S)
		return np.sqrt(np.mean((y-A)**2))

		
