import pickle
import numpy as np
import utils

from collections import deque

class SimileDataManipulator():

	def __init__(self, data_param, simile_param, op_param):
		self.data_param = data_param
		self.simile_param = simile_param
		self.op_param = op_param

	def load_train_episodes (self, config):
		# Reading Data File Paths
		train_file = open(config['DATA']['train_file'], 'r')
		train_file_paths = train_file.readlines()
		n_train_episodes = len(train_file_paths)
		train_file.close()
		print ('Number of Training Episodes: ', n_train_episodes)

		try: 
			valid_file = config['DATA']['valid_file']
			include_validation = True
		except:
			include_validation = False

		if (include_validation):
			valid_file = open(config['DATA']['valid_file'], 'r')
			valid_file_paths = valid_file.readlines()
			n_valid_episodes = len(valid_file_paths)
			valid_file.close()
			print ('Number of Validation Episodes: ', n_valid_episodes) 

		# Formating Episodes
		X_train_episodes, Y_train_episodes, init_values_train = self.format_episodes (train_file_paths, with_gt=True)
		if (include_validation):
			X_valid_episodes, Y_valid_episodes, init_values_valid = self.format_episodes (valid_file_paths, with_gt=True)

		if (self.data_param['init_pred'] == "from_data"):
			self.data_param.update({'init_pred_train' : init_values_train})
		else:
			self.data_param.update({'init_pred_train' : [self.data_param['init_pred']]*len(X_train_episodes)})

		if (include_validation):
			if (self.data_param['init_pred'] == "from_data"):
				self.data_param.update({'init_pred_valid' : init_values_valid})
			else:
				self.data_param.update({'init_pred_valid' : [self.data_param['init_pred']]*len(X_valid_episodes)})

		data = {'X_train_episodes' : X_train_episodes, 'Y_train_episodes' : Y_train_episodes}
		if (include_validation):
			data.update({'X_valid_episodes' : X_valid_episodes, 'Y_valid_episodes' : Y_valid_episodes})
		
		self.op_param.update({'include_validation' : include_validation})

		assert(len(data['X_train_episodes']) == len(data['X_train_episodes']))
		if (self.op_param['include_validation']):
			assert(len(data['X_valid_episodes']) == len(data['Y_valid_episodes']))

		# Taking number of datapoints per episode
		n_dp_epsd_train = [len(episode) for episode in data['X_train_episodes']]
		self.data_param.update({'n_dp_epsd_train' : n_dp_epsd_train})
		if (self.op_param['include_validation']):
			n_dp_epsd_test = [len(episode) for episode in data['X_valid_episodes']]
			self.data_param.update({'n_dp_epsd_test' : n_dp_epsd_test})

		return data, self.data_param, self.op_param


	def load_test_episodes (self, config, training_param):
		# Reading Data File Paths
		test_file = open(config['DATA']['test_file'], 'r')
		test_file_paths = test_file.readlines()
		n_test_episodes = len(test_file_paths)
		test_file.close()

		print ('Number of Test Episodes: ', n_test_episodes)

		if (config['DATA']['with_gt'] == 'True'):
			X_test_episodes, Y_test_episodes, init_values_test = self.format_episodes (test_file_paths, with_gt=True)
			data = { 'X_test_episodes' : X_test_episodes, 'Y_test_episodes' : Y_test_episodes }
			include_gt = True

			if (self.data_param['init_pred'] == "from_data"):
				self.data_param.update({'init_pred_test' : init_values_test})
			else:
				self.data_param.update({'init_pred_test' : [self.data_param['init_pred']]*len(X_test_episodes)})
		else:
			X_test_episodes = self.format_episodes (test_file_paths, with_gt=False)
			data = { 'X_test_episodes' : X_test_episodes }
			include_gt = False

			if (self.data_param['init_pred'] == "from_data"):
				print ('Error: Cannot take initial values from data since ground truth was not provided. ')
				print ('Please specify the initial value to the sequence on the config file. ')
				print ('You may check the config file example available on the repo for reference.' )
				exit()
			else:
				self.data_param.update({'init_pred_test' : [self.data_param['init_pred']]*len(X_test_episodes)})

		self.op_param.update({'include_gt' : include_gt})

		# Taking number of datapoints per episode
		n_dp_epsd_test = [len(episode) for episode in data['X_test_episodes']]
		self.data_param.update({'n_dp_epsd_test' : n_dp_epsd_test})

		return data, self.data_param, self.op_param


	def format_episodes (self, path_list, with_gt):
		X_episodes = []

		if (with_gt == True):
			Y_episodes = []
			init_values = []

		for path in path_list:
			path = path.split('\n')[0]
			data = pickle.load(open(path, 'rb'))

			try:
				data.shape
				data.shape[0]
				data.shape[1]
			except:
				print ('Error: File ', path, ' with shape ', data.shape, 'is not in the correct format.')
				print ('Please check the repo documentation for instructions on the expected format for the data. ')
				exit()

			if (with_gt == True):
				if (data.shape[1] != self.data_param['n_targets'] + self.data_param['n_features']):
					print ('Error: File ', path, ' with shape ', data.shape, 'does not seem to contain ', self.data_param['n_features'], 
						   'feature variables and ', self.data_param['n_targets'], 'target variables, as specified in file "config_train.py"')
					print ('Please check the repo documentation for instructions on the expected format for the data. ')
					exit()
			else:
				if (data.shape[1] != self.data_param['n_features']):
					print ('Error: File ', path, ' with shape ', data.shape, 'does not seem to contain ', self.data_param['n_features'], 
						   'feature variables as specified in the config file')
					print ('Please check the repo documentation for instructions on the expected format for the data. ')
					exit()

			X_episodes.append(data[:, :self.data_param['n_features']])
			
			if (with_gt == True):
				Y_episodes.append(data[:, self.data_param['n_features']:])
				init_values.append(data[0, self.data_param['n_features']:])
				
		if (with_gt == True):	
			assert (len(X_episodes) == len(Y_episodes) == len(path_list))

			return X_episodes, Y_episodes, init_values

		else:
			
			assert (len(X_episodes) == len(path_list))
			return X_episodes


	def normalize_train(self, data):
		# Taking number of datapoints per episode
		self.n_datapoints_episd_train = [len(episode) for episode in data['X_train_episodes']]
		if (self.op_param['include_validation']):
			self.n_datapoints_episd_test = [len(episode) for episode in data['X_valid_episodes']]

		if (self.op_param['normalize_input'] == 'True'):
			# Normalizing data to be in range (-0.5, +0.5)
			X_train_episodes, X_norm_param = self.normalize_train_episodes (data['X_train_episodes'], env=True)
			data.update({'X_train_episodes' : X_train_episodes})

			if (self.op_param['include_validation']):
				X_valid_episodes = self.normalize_test_episodes (data['X_valid_episodes'], X_norm_param, env=True)
				data.update({'X_valid_episodes' : X_valid_episodes})	

		if (self.op_param['normalize_output'] == 'True'):
			# Normalizing data to be in range (-0.5, +0.5)
			Y_train_episodes, Y_norm_param = self.normalize_train_episodes(data['Y_train_episodes'], env=False)
			data.update({'Y_train_episodes' : Y_train_episodes})

			self.data_param['init_pred_train'] = self.normalize_init(self.data_param['init_pred_train'], Y_norm_param)
			# self.data_param.update({'init_pred_train' : self.data_param['init_pred_train']})

			if (self.op_param['include_validation']):
				Y_valid_episodes = self.normalize_test_episodes (data['Y_valid_episodes'], Y_norm_param, env=False)
				data.update({'Y_valid_episodes' : Y_valid_episodes})

				self.data_param['init_pred_valid'] = self.normalize_init(self.data_param['init_pred_valid'], Y_norm_param)

		return data, self.data_param

	def normalize_init (self, init_values, Y_norm_param):
		norm_init = []
		for init in init_values:
			norm_init.append(utils.normalize_test_data(np.array(init).reshape(1,-1), 
							 Y_norm_param['min_val'], Y_norm_param['max_val'])[0])
		return norm_init
		
	def normalize_test (self, data):
		self.n_datapoints_episd_test = [len(episode) for episode in data['X_test_episodes']]

		if (self.op_param['normalize_input'] == 'True'):
			# Normalizing data to be in range (-0.5, +0.5)
			X_norm_param = pickle.load(open(self.data_param['model_dir'] + self.data_param['model_label'] + '_Xnorm.p', 'rb'))
			X_norm_param['min_val'] = [float(i) for i in X_norm_param['min_val']]
			X_norm_param['max_val'] = [float(i) for i in X_norm_param['max_val']]

			X_test_episodes = self.normalize_test_episodes (data['X_test_episodes'], X_norm_param, env=True)
			data.update({'X_test_episodes' : X_test_episodes})

		if (self.op_param['normalize_output'] == 'True'):
			Y_norm_param = pickle.load(open(self.data_param['model_dir'] + self.data_param['model_label'] + '_Ynorm.p', 'rb'))

			self.data_param['init_pred_test'] = self.normalize_init(self.data_param['init_pred_test'], Y_norm_param)

			if ('Y_test_episodes' in data):
				Y_test_episodes = self.normalize_test_episodes(data['Y_test_episodes'], Y_norm_param, env=False)
				if (self.op_param['include_gt'] == True):
					data.update({'Y_test_episodes' : Y_test_episodes})

		return data, self.data_param

	def normalize_train_episodes (self, episodes, env):

			episodes_array = np.concatenate(episodes, axis=0)
			episodes_array, min_val, max_val = utils.normalize_train_data (episodes_array)

			assert(episodes_array.shape[0] == np.sum(self.n_datapoints_episd_train))
			if (env == True):
				assert (episodes_array.shape[1] == self.data_param['n_features'])
			else:
				assert (episodes_array.shape[1] == self.data_param['n_targets'])

			# Converting back to list of episodes
			if (len(self.n_datapoints_episd_train) > 1):
				list_idx, prev_sum = [], 0
				for n_dp in self.n_datapoints_episd_train[:-1]:
					prev_sum += n_dp
					list_idx.append(prev_sum)

				episodes = np.split(episodes_array, list_idx)
			else:
				episodes = [episodes_array]

			# Making sure data looks correct before proceeding
			assert (len(episodes) == len(self.n_datapoints_episd_train))
			for i, episode in zip(range(len(episodes)), episodes):
				assert (episode.shape[0] ==  self.n_datapoints_episd_train[i])

			# Saving Normalization Parameters
			norm_param = { 'min_val' : min_val, 'max_val' : max_val }
			if (env == True):
				pickle.dump(norm_param, open(self.data_param['output_dir'] + self.data_param['model_label'] + '_Xnorm.p', 'wb'))
			else:
				pickle.dump(norm_param, open(self.data_param['output_dir'] + self.data_param['model_label'] + '_Ynorm.p', 'wb'))

			return episodes, norm_param

	def normalize_test_episodes (self, episodes, norm_param, env):

		min_val, max_val = norm_param['min_val'], norm_param['max_val']

		episodes_array = np.concatenate(episodes, axis=0)
		episodes_array = utils.normalize_test_data (episodes_array, min_val, max_val)

		assert(episodes_array.shape[0] == np.sum(self.n_datapoints_episd_test))
		if (env == True):
			assert(episodes_array.shape[1] == self.data_param['n_features'])
		else:
			assert(episodes_array.shape[1] == self.data_param['n_targets'])

		# Converting back to list of episodes
		if (len(self.n_datapoints_episd_test) > 1):
			list_idx, prev_sum = [], 0
			for n_dp in self.n_datapoints_episd_test[:-1]:
				prev_sum += n_dp
				list_idx.append(prev_sum)

			episodes = np.split(episodes_array, list_idx)
		else:
			episodes = [episodes_array]

		# Making sure data looks correct before proceeding
		assert (len(episodes) == len(self.n_datapoints_episd_test))
		for i, episode in zip(range(len(episodes)), episodes):
			assert (episode.shape[0] == self.n_datapoints_episd_test[i])

		return episodes


	def build_input_structure_with_lookfuture (self, X_episodes):
		if (self.simile_param['tao'] % 2 == 0):
			mid = int(0.5*(self.simile_param['tao']))	# Even tao
		else: 
			mid = int(0.5*(self.simile_param['tao']-1))	# Odd tao

		X_all = []
		for X in X_episodes:
			# Resetting deque for new episode
			X_context = deque()
			n_datapoints = len(X)

			## Initializing Context Structure for current episode ##
			# Initializing past context: replicate features on fist frame 'mid' times to model past actions plus '1' time corresponding to the 'current' frame
			for i in range(mid+1):
				X_context.append(X[0])
			# Initialize future context
			t = 1
			for i in range(mid+1, self.simile_param['tao']):
				X_context.append(X[t])
				t += 1

			features = []
			features.append(np.hstack(X_context)) # Appending initial structure

			# Going over every element in current episode -- starting from position 't'
			for feat in X[t:]:
				# Remove last element (not part of context anymore)	
				X_context.popleft()
				# Add frame feature from  't' steps in the future to datastructure
				X_context.append(feat)
				# Flatten deque structure
				feature_array = np.hstack(X_context)

				assert (feature_array.shape[0] == self.simile_param['tao']*self.data_param['n_features'])

				features.append(feature_array)

			# Padding end of the episode with context from last frame
			for i in range(t-1):
				X_context.popleft()
				X_context.append(X[-1])
				feature_array = np.hstack(X_context)
				assert (feature_array.shape[0] == self.simile_param['tao']*self.data_param['n_features'])
				features.append(feature_array)

			assert (len(features) == n_datapoints)

			X_all.append(features)

		return np.concatenate(X_all, axis=0)


	def build_input_structure (self, X_episodes):
		X_all = []

		for X in X_episodes:
			# Resetting deque for new episode
			X_hist = deque()

			# Initialize feature by replicating 'tao' times the features on first frame
			for t in range(self.simile_param['tao']):  
				X_hist.append(X[0])

			# Going over all elements in current episode
			features = []
			for feat in X:
				# Remove last element (not part of context anymore)	
				X_hist.popleft()	
				# Add current frame feature to datastructure
				X_hist.append(feat)
				# Flatten deque structure
				feature_array = np.hstack(X_hist)

				assert (feature_array.shape[0] == self.simile_param['tao']*self.data_param['n_features'])

				features.append(feature_array)

			X_all.append(features)				

		return np.concatenate(X_all, axis=0)


	def build_init_action_structure (self, Y_episodes):
		A_hist = []
		A_gt = []

		for Y in Y_episodes:
			n_frames = 0
			# Resetting deque for new episode
			recent_actions = deque()

			for t in range(self.simile_param['tao']):  # Keep track of 'tao' previous actions plus current action
				recent_actions.append(Y[0])

			action_hist = []  # Contains action history from single episode
			action_gt = []

			for action in Y:
				# Remove last element (not part of context anymore)	
				recent_actions.popleft()	
				# Add current action to datastructure - this is (at), while the rest of structure is (at-1:t-tao)
				recent_actions.append(action)
				# Flatten deque structure
				action_array = np.hstack(recent_actions)
				
				assert (action_array.shape[0] == self.data_param['n_targets']*self.simile_param['tao'])

				# History part of the structure (at-1:at-tao)
				action_hist.append(action_array[:self.data_param['n_targets']*(self.simile_param['tao']-1)])
				# Current action (at)
				action_gt.append(action_array[self.data_param['n_targets']*(self.simile_param['tao']-1):])

			A_hist.append(action_hist) # Contains action history from all episodes
			A_gt.append(action_gt)

		return np.concatenate(A_hist, axis=0), np.concatenate(A_gt, axis=0)

	def get_expert_derivatives(self, A_gt):
		expert_derivatives = np.zeros((A_gt.shape), dtype='float64')
		for i in range(1, A_gt.shape[0]):
			expert_derivatives[i] = A_gt[i] - A_gt[i-1]
		return expert_derivatives

