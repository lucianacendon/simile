from sklearn import linear_model
import numpy as np

class Autoregressor():
	def __init__ (self):
		self.predictor = None
		self.function_type = None

	def set_linear(self, lambda_autorgr, action_hist, action_target):
		self.function_type = "linear"
		self.predictor = linear_model.Ridge(alpha = lambda_autorgr).fit(action_hist, action_target)

	def set_constant(self):
		self.function_type = "constant"

	def set_average(self):
		self.function_type = "average"

	def set_geometric_velocity(self, gamma):
		self.gamma = gamma
		self.function_type = "geometric_velocity"

	def predict(self, state):
		assert(self.function_type != None)
		if (self.function_type == "constant"):
			return state[:, -1]
		elif(self.function_type == "average"):
			return np.mean(state)
		elif(self.function_type == "linear"):
			return self.predictor.predict(state)
		elif(self.function_type == "geometric_velocity"):
			return self.predict_geometric_velocity(state)
		else:
			print ("Error: Invalid autoregressor type", self.function_type)
			exit()

	def predict_geometric_velocity (self, state):
		for j in range(state.shape[0]):
			offset = state[j, -1] 
			velocities = 0
			acc = 0
			curr_mult = self.gamma

			for i in range(1, state.shape[1]):
				k = i + 1
				
				velocities += (state[j, -i] - state[j, -k])*curr_mult
				acc += curr_mult
				curr_mult *= self.gamma

			output = (offset + float(velocities/acc))

		return output
