import tensorflow as tf
import numpy as numpy
import pickle
import matplotlib.pyplot as plt
import graph_utils

class NeuralNet():

	def __init__ (self, input_shape, data_param, model_param, learning_param):
		
		self.data_param = data_param
		self.model_param = model_param
		self.input_shape = input_shape
		self.session = model_param['session']

		if (model_param['train'] == True):
			# Training Mode #
			self.it = model_param['it']
			self.learning_param = learning_param
			self.log_file = data_param['log_file']
			self.policy_file = self.data_param['output_dir'] + self.data_param['model_label'] + '_NN_policy_' + str(self.it) + '.ckpt'
			self.graph = graph_utils.SimileGraphUtils(data_param, None)
			
			self.build_graph()
		else:
			# Test mode #
			self.policy_file = self.model_param['policy_file']

			self.build_network()

		
	def build_network (self):
		self.rate = tf.placeholder(tf.float32, shape=(), name='dropout_rate')
		self.inputs = tf.placeholder(tf.float32, shape=(None, self.input_shape), name = 'inputs')
		self.y = tf.placeholder(tf.float32, shape=(None, self.data_param['n_targets']), name = 'actions')

		# First Layer
		w1 = tf.Variable(tf.random_normal([self.model_param['hidsize_1'], self.input_shape], stddev = 0.01), name = 'w1')
		b1 = tf.Variable(tf.constant(0.1, shape = (self.model_param['hidsize_1'], 1)), name = 'b1')
		y1 = tf.nn.relu(tf.add(tf.matmul(w1, tf.transpose(self.inputs)), b1))
		y1 = tf.nn.dropout(y1, rate=self.rate)

		# Second Layer
		w2 = tf.Variable(tf.random_normal([self.model_param['hidsize_2'], self.model_param['hidsize_1']], stddev = 0.01), name = 'w2')
		b2 = tf.Variable(tf.constant(0.1, shape=(self.model_param['hidsize_2'], 1)), name = 'b2')
		y2 = tf.nn.relu(tf.add(tf.matmul(w2, y1), b2))
		y2 = tf.nn.dropout(y2, rate=self.rate)

		# Output layer
		wo = tf.Variable(tf.random_normal([self.data_param['n_targets'], self.model_param['hidsize_2']], stddev = 0.01), name = 'wo')
		bo = tf.Variable(tf.random_normal([self.data_param['n_targets'], 1]), name = 'bo')

		self.y_hat = tf.transpose(tf.add(tf.matmul(wo, y2), bo))
		self.reg = tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2) + tf.nn.l2_loss(wo)

		self.loss = tf.reduce_mean(tf.square(self.y - self.y_hat) + self.model_param['regularization']*self.reg) # loss + beta * regularizers

		self.error = tf.sqrt(tf.reduce_mean(tf.square(self.y - self.y_hat)))

		self.saver = tf.train.Saver()

	def build_graph (self):

		self.build_network()

		self.optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_param['learning_rate']).minimize(self.loss)


	def train(self, States, Actions):

		self.session.run(tf.global_variables_initializer())
		S_train = States['S_train']
		A_train = Actions['A_train']
		n_datapoints = S_train.shape[0]

		if ('S_valid' in States.keys()) and ('A_valid' in Actions.keys()):
			include_validation = True
			S_valid = States['S_valid']
			A_valid = Actions['A_valid']
		else:
			include_validation = False

		print ("-------- Policy {:1d} --------".format(self.it))
		self.log_file.writelines("-------- Policy {:1d} --------\n".format(self.it))

		n_batches = int(n_datapoints / self.learning_param['batch_size'])

		progress_data = {'train_errors' : [], 'measured_epoches' : [], 'costs' : []}
		if (include_validation):
			progress_data.update({'valid_errors' : []})

		for epoch in range(1, self.learning_param['n_epoches']+1):
			avg_cost = 0.0
			for i in range(n_batches+1): 
				batch_x = S_train[i*self.learning_param['batch_size'] : (i+1)*self.learning_param['batch_size'], :]
				batch_y = A_train[i*self.learning_param['batch_size'] : (i+1)*self.learning_param['batch_size'], :]
				
				_, c = self.session.run([self.optimizer, self.loss], feed_dict = {self.inputs: batch_x, 
																				  self.y: batch_y, 
																				  self.rate: 0.5})
				avg_cost += c

			avg_cost /= n_datapoints
			progress_data['costs'].append(avg_cost)

			if (epoch % self.learning_param['save_freq'] == 0) or (epoch == 1):
				# Saving Model files
				self.saver.save(self.session, self.policy_file)

				# Calculating errors
				train_err = self.error.eval(feed_dict={self.inputs: S_train, self.y: A_train, self.rate: 0})
				progress_data['train_errors'].append(train_err)
				progress_data['measured_epoches'].append(epoch)

				print ("Epoch: {:f}, Train Cost: {:f} ".format(epoch, avg_cost))
				print ("Train Error: {:f} ".format(train_err))
				self.log_file.writelines("Epoch: {:d}, Train Cost: {:f} \n".format(epoch, avg_cost))
				self.log_file.writelines("Train Error: {:f} \n".format(train_err))

				if (include_validation):
					valid_err = self.error.eval(feed_dict={self.inputs: S_valid, self.y: A_valid, self.rate: 0})
					progress_data['valid_errors'].append(valid_err)

					print ("Validation Error: {:f} ".format(valid_err))
					self.log_file.writelines("Validation Error: {:f} \n".format(valid_err))
					
				self.graph.plot_nn_training_progress(progress_data, epoch, self.it)


	def get_raw_prediction (self, state):
		self.saver.restore(self.session, self.policy_file)
		return self.y_hat.eval(feed_dict={self.inputs: state, self.rate: 0})
       


	