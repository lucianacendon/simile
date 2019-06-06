import numpy as np
import matplotlib.pyplot as plt
import pickle
import utils

class SimileGraphUtils():

	def __init__ (self, data_param, op_param):
		self.data_param = data_param
		self.op_param = op_param

	def set_fig_size (self, n_datapoints):

		if (0.002*n_datapoints > 6.4):
			if (0.002*n_datapoints < 200):
				x_size = 0.002*n_datapoints
			else:
				x_size = 200
		else:
			x_size = 6.4 # matplotlib default

		if (3*self.data_param['n_targets'] > 4.8):
			if (3*self.data_param['n_targets'] < 200):
				y_size = 3*self.data_param['n_targets']
			else:
				y_size = 200
		else:
			y_size = 4.8 # matplotlib default

		return x_size, y_size

	def plot_results (self, predictions, ground_truth, datatype, policy_n, flag):
		if (flag == 'check_env_feat'):
			label = 'Prediction'
		elif (flag == 'rollout'):
			label = 'Rollout'
		else:
			print ('Invalid flag in plot_results function', flag)
			exit()

		x = np.arange(predictions.shape[0])

		(x_size, y_size) = self.set_fig_size(predictions.shape[0])
		plt.figure(figsize=(x_size, y_size))

		plt.title ('Prediction vs. G.T')
		for i in range(1, self.data_param['n_targets']+1):
			ax = plt.subplot(self.data_param['n_targets'], 1, i)
			if (not ground_truth is None):
				plt.plot(x, predictions[:, i-1], 'b-', ground_truth[:, i-1], 'g-', linewidth=0.5)
			else:
				plt.plot(x, predictions[:, i-1], 'b-', linewidth=0.5)

			plt.ylabel(self.data_param['target_labels'][i-1])

		ax.legend([str(label), 'Ground Truth'], loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2)
		plt.xlabel('Time')

		if (datatype != 'test') and (datatype != 'denorm'):
			plt.savefig(self.data_param['log_dir'] + self.data_param['model_label'] + '_' + str(label) + '_' + 
						datatype + '_policy_' + str(policy_n) + '.eps', format='eps')
		else:
			plt.savefig(self.data_param['log_dir'] + self.data_param['model_label'] + '_' + str(label) + '_' + 
						datatype + '_pol' + str(policy_n) + '.eps', format='eps')
		plt.close('all')

	def plot_autoregressor (self, predictions, autoregressor, raw_pred, ground_truth, datatype, policy_n):

		x = np.arange(predictions.shape[0])

		(x_size, y_size) = self.set_fig_size(predictions.shape[0])
		plt.figure(figsize=(x_size, y_size))

		plt.title('Raw Prediction vs. Rollout vs. Autoregressor vs. Ground Truth')

		for i in range(1, self.data_param['n_targets']+1):
			ax = plt.subplot(3, 1, i)
			if (not ground_truth is None):
				plt.plot(x, raw_pred[:, i-1], 'k-', predictions[:, i-1], 'b-', autoregressor[:, i-1], 'r-', 
						 ground_truth[:, i-1], 'g-', linewidth=0.5)
			else:
				plt.plot(x, raw_pred[:, i-1], 'k-', predictions[:, i-1], 'b-', 
							autoregressor[:, i-1], 'r-', linewidth=0.5)
			plt.ylabel(self.data_param['target_labels'][i-1])
		
		plt.xlabel('Time')
		ax.legend(['Raw Prediction', 'Rollout', 'Autoreg', 'Ground Truth'], 
			      loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=4)

		if (datatype != 'test') and (datatype != 'denorm'):
			plt.savefig(self.data_param['log_dir'] + self.data_param['model_label'] + '_Autoreg_' + 
				        datatype + '_policy_' + str(policy_n) + '.eps', format='eps')
		else:
			plt.savefig(self.data_param['log_dir'] + self.data_param['model_label'] + '_Autoreg_' + 
						datatype + '_pol' + str(policy_n) + '.eps', format='eps')

		plt.close('all')

	def plot_smoothfeedback (self, predictions, smooth_feedback, ground_truth, datatype, policy_n):
		
		x = np.arange(predictions.shape[0])

		(x_size, y_size) = self.set_fig_size(predictions.shape[0])
		plt.figure(figsize=(x_size, y_size))

		plt.title('Rollout vs. Smooth Feedback vs. G.T')
		for i in range(1, self.data_param['n_targets']+1):
			ax = plt.subplot(self.data_param['n_targets'], 1, i)
			if (not ground_truth is None):
				plt.plot(x, predictions[:, i-1], 'b-', smooth_feedback[:, i-1], 'r-',  
						    ground_truth[:, i-1], 'g-', linewidth=0.5)
			else:
				plt.plot(x, predictions[:, i-1], 'b-', smooth_feedback[:, i-1], 'r-', linewidth=0.5)

			plt.ylabel(self.data_param['target_labels'][i-1])

		ax.legend(['Rollout', 'Smooth Feedback', 'Ground Truth'], loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3)
		plt.xlabel('Time')

		plt.savefig(self.data_param['log_dir'] + self.data_param['model_label'] + '_SFBCK_' + 
					datatype + '_policy_' + str(policy_n) + '.eps', format='eps')
		plt.close('all')

	def plot_nn_training_progress (self, data, epoch_n, it):
		
		plt.figure()
		plt.subplot(1, 2, 1)
		x_axis = np.arange(epoch_n)
		plt.plot(x_axis, data['costs'])
		plt.xlabel('Epoch')
		plt.ylabel('Cost')

		plt.subplot(1, 2, 2)
		plt.plot(data['measured_epoches'], data['train_errors'], 'b-', label='Train Error')
		if ('valid_errors' in data.keys()):
			plt.plot(data['measured_epoches'], data['valid_errors'], 'r-', label='Valid Error')
		plt.xlabel('Epoch')
		plt.ylabel('Error')
		plt.legend()

		plt.savefig(self.data_param['log_dir'] + self.data_param['model_label'] + 
			        '_cost_error_policy_' + str(it) + '.eps', format='eps')

		plt.close('all')

	def plot_rollout_train (self, rollout, Actions, it):

		if (self.op_param['plot_train_results'] == 'True'):
			if (self.op_param['only_env_feat'] == 'True'):
				self.plot_results (rollout['y_hat_raw_train'], Actions['A_train_gt'], 'train', it, 'check_env_feat')
			else:
				self.plot_results (rollout['y_hat_train'], Actions['A_train_gt'], 'train', it, 'rollout')
				self.plot_autoregressor (rollout['y_hat_train'], rollout['a_h_train'], rollout['y_hat_raw_train'], 
					                     Actions['A_train_gt'], 'train', it)

		if (self.op_param['include_validation']):
			if (self.op_param['plot_valid_results'] == 'True'):
				if (self.op_param['only_env_feat'] == 'True'):
					self.plot_results (rollout['y_hat_raw_valid'], Actions['A_valid_gt'],'valid', it, 'check_env_feat')
				else:
					self.plot_results (rollout['y_hat_valid'], Actions['A_valid_gt'], 'valid', it, 'rollout')
					self.plot_autoregressor (rollout['y_hat_valid'], rollout['a_h_valid'], rollout['y_hat_raw_valid'], 
						                     Actions['A_valid_gt'], 'valid', it)


	def plot_rollout_test (self, rollout):
		# Note: A_gt was passed to this class during "SIMILE_PREDICT" initialization if present

		if (self.op_param['normalize_output'] == 'True'):
			Y_norm_param = pickle.load(open(self.data_param['model_dir'] + self.data_param['model_label'] + '_Ynorm.p', 'rb'))

		if (self.op_param['normalize_output'] == 'True'):
			if (self.op_param['include_gt'] == True):
				A_gt_denorm = utils.denormalize(self.A_gt, Y_norm_param['min_val'], Y_norm_param['max_val'])
			else:
				A_gt_denorm = None

		if (self.op_param['only_env_feat'] == 'True'):
			self.plot_results (rollout['y_hat_raw'], self.A_gt, 'test', self.pol_load-1, 'check_env_feat')				
			if (self.op_param['normalize_output'] == 'True'):
				rollout_raw_denorm = utils.denormalize(rollout['y_hat_raw'], Y_norm_param['min_val'], Y_norm_param['max_val'])
				self.plot_results (rollout_raw_denorm, A_gt_denorm, 'denorm',  self.pol_load-1, 'check_env_feat')
		else:
			self.plot_results (rollout['y_hat'], self.A_gt, 'test',  self.pol_load-1, 'rollout')
			self.plot_autoregressor (rollout['y_hat'], rollout['a_h'], rollout['y_hat_raw'], self.A_gt, 'test', self.pol_load-1)

			if (self.op_param['normalize_output'] == 'True'):
				rollout_denorm = utils.denormalize(rollout['y_hat'], Y_norm_param['min_val'], Y_norm_param['max_val'])
				self.plot_results (rollout_denorm, A_gt_denorm, 'denorm', self.pol_load-1, 'rollout')


