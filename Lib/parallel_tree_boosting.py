import xgboost as xgb
import numpy as np
import pickle

class XGBoost ():

	def __init__(self, data_param, model_param, learning_param):
		self.data_param = data_param
		self.model_param = model_param

		if (model_param['train'] == False):
			self.xgreg = self.load_model_file(model_param['policy_file'])
		else:
			self.it = model_param['it']
			self.learning_param = learning_param
			self.log_file = data_param['log_file']

	def train (self, States, Actions):
		self.lr = self.learning_param['learning_rate']

		S_train = States['S_train']
		A_train = Actions['A_train']

		if ('S_valid' in States.keys()) and ('A_valid' in Actions.keys()):
			self.include_validation = True
			S_valid = States['S_valid']
			A_valid = Actions['A_valid']
		else:
			self.include_validation = False

		print ("-------- Policy {:1d} --------".format(self.it))
		self.log_file.writelines("-------- Policy {:1d} --------\n".format(self.it))

		self.xgreg = []
		for i in range (0, self.data_param['n_targets']):
			print ('Training ' + str(self.data_param['target_labels'][i]) + '...')
			rgr = xgb.XGBRegressor (objective ='reg:squarederror', learning_rate = float(self.lr), 
				                    reg_lambda = float(self.model_param['regularization']), 
				                    n_estimators = int(self.model_param['n_estimators'])) 
			self.xgreg.append(rgr.fit(S_train, A_train[:, i]))

		train_error = self.error_raw_current_model (S_train, A_train)
		print ('Training Error: ', train_error)
		self.log_file.writelines("Training Error: {:f} \n".format(train_error))	
		if (self.include_validation):
			valid_error = self.error_raw_current_model (S_valid, A_valid)
			print ('Validation Error: ', valid_error)
			self.log_file.writelines("Validation Error: {:f}\n".format(valid_error))	
		
		pickle.dump(self.xgreg, open(self.data_param['output_dir'] + self.data_param['model_label'] + 
					'_Xboost_policy' + str(self.it) + '.p', 'wb'))


	def load_model_file (self, policy_file):
		return pickle.load(open(policy_file, 'rb'))

	def get_raw_prediction (self, state):
		pred = np.zeros((state.shape[0], self.data_param['n_targets']))

		for i in range(0, pred.shape[1]):
			pred[:, i] = self.xgreg[i].predict(state).reshape(1, -1)
		return pred

	def error_raw_current_model (self, S, A):
		y = self.get_raw_prediction (S)
		return np.sqrt(np.mean((y-A)**2))


