import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

#assume all input x in activation function are numpy array
def relu(x,derivative=False):
	if derivative:
		res = x.copy()
		res[res<=0] = 0
		res[res>0] = 1
		return res
	return np.maximum(0,x)

def sigmoid(x,derivative=False):
	p = 1/(1 + np.exp(-x))
	if derivative:
		return p*(1-p)
	return p

def softplus(x,derivative=False):
	if derivative:
		return sigmoid(x)
	return np.log(1 + np.exp(x))

def linear(x,derivative=False):
	if derivative:
		res = x.copy()
		res[:] = 1
		return res
	return x

def return_nearest(x,unique_vals):
	res = x.copy()
	for u in unique_vals:
		res[np.abs(res)]


def random_matrix(m,n):
	return np.random.randn(m,n)

activation_dict = {
	'linear' : linear,
	'sigmoid' : sigmoid,
	'softplus' : softplus,
	'relu' : relu
}

class LengthError(Exception):
	pass

class NotBinaryClassificationError(Exception):
	pass

class LabelNotEncodedError(Exception):
	pass

class Layer:
	def __init__(self,n_neuron,activation_function='linear'):
		self.n_neuron = n_neuron
		self.activation_function = activation_dict[activation_function]
		self.out = None
		self.input = None
		self.init_weight = None
		self.weights = None
		self.biases = np.zeros(n_neuron)

	def forward_pass(self,vals):
		self.input = vals
		self.out = self.activation_function(vals)
		return self.out

	def reset(self):
		self.biases = np.array([np.zeros(self.n_neuron)])
		if self.init_weight is not None:
			self.weights = self.init_weight.copy()

class ANNClassifier:
	def __init__(self,random_state=None):
		self.layers = list()
		self.learning_rate = None
		self.epoch = None
		self.errors = None
		self.error_per_epoch = None
		self.mini_batch = None
		np.random.seed(random_state)

	def reset(self):
		self.errors = list()
		self.error_per_epoch = list()
		for l in self.layers:
			l.reset()

	def compile(self,learning_rate,mini_batch,epoch):
		self.learning_rate = learning_rate
		self.epoch = epoch
		self.mini_batch = mini_batch

	def add(self,layer):
		self.layers.append(layer)
		if len(self.layers) != 1:
			m = self.layers[-2].n_neuron
			n = self.layers[-1].n_neuron
			self.layers[-1].weights = random_matrix(m,n)
			self.layers[-1].init_weight = self.layers[-1].weights.copy()

	def forward_propagate(self,input_value):
		values = self.layers[0].forward_pass(input_value)
		for l in range(1,len(self.layers)):
			values = np.dot(values,self.layers[l].weights) + self.layers[l].biases
			values = self.layers[l].forward_pass(values)
		return values

	def predict(self,input_value):
		res = self.forward_propagate(input_val)

	def ssr(self,output,expected,derivative=False):
		res = None
		if not derivative:
			res = 0.5* (output - expected)**2
		else:
			res = (output - expected)
		return res

	def mse(self,output,expected,derivative=False):
		return np.mean(self.ssr(output,expected,derivative),axis=0).mean()

	def chain_rule(self,expected,var='weight',reverse_layer=0):
		dCdOut = self.ssr(self.layers[-1].out,expected,derivative=True)
		res = dCdOut
		for i in range(reverse_layer):
			dOdI = self.layers[-1-i].activation_function(self.layers[-1-i].input,derivative=True)
			res = res * dOdI
			dIdO = self.layers[-1-i].weights
			res = np.dot(res,dIdO.T)
		idx= -1 - reverse_layer
		dOdI = self.layers[idx].activation_function(self.layers[idx].input,derivative=True)
		res = res * dOdI
		if var == 'weight':
			dIdW = self.layers[idx-1].out
			res = np.dot(dIdW.T,res)/len(expected)
			return res
		elif var == 'bias':
			return np.mean(res,axis=0)

	def back_propagate(self,expected,lr):
		for i in range(1,len(self.layers)):
			self.layers[i].weights = self.layers[i].weights - (self.chain_rule(expected,var='weight',reverse_layer=len(self.layers)-1-i) * lr)
			self.layers[i].biases = self.layers[i].biases - (self.chain_rule(expected,var='bias',reverse_layer=len(self.layers)-1-i) * lr)

	def SGD(self,input_val,expected,lr):
		output = self.forward_propagate(input_val)
		self.back_propagate(expected,lr)
		return self.mse(output,expected)

	def fit(self,x,y):
		if len(x) != len(y):
			raise LengthError("Inconsistent number of datapoint (feature : {0}, target : {1})".format(len(x),len(y)))
		if len(x.columns) != self.layers[0].n_neuron:
			raise LengthError("Invalid number of initial neurons (feature column : {0} columns, initial neurons : {1} neurons)".format(len(x.columns),self.layers[0].n_neuron))
		if len(y.unique()) != 2 :
			raise NotBinaryClassificationError("Not a binary classification problem (number of label's class : {0})".format(len(y.unique)))
		if self.layers[-1].n_neuron != 1:
			raise NotBinaryClassificationError("Output layer should be consist of 1 neuron (current number of output neuron : {0})".format(self.layers[-1].n_neuron))
		if 1 not in y.unique() or 0 not in y.unique():
			raise LabelNotEncodedError("Label column should be encoded into 0s and 1s")

		self.reset()
		iteration = int(np.ceil(len(x)/self.mini_batch))
		for e in range(self.epoch):
			start = time.time()
			total_loss = 0
			for i in range(iteration):
				input_val = x.iloc[i*self.mini_batch:(i+1)*self.mini_batch].values
				expected = np.array([y.iloc[i*self.mini_batch:(i+1)*self.mini_batch].values]).T
				loss = self.SGD(input_val,expected,self.learning_rate)
				total_loss += loss
				self.errors.append(loss)
			loss_per_epoch = total_loss/iteration
			print("Epoch {0} | Loss per Epoch : {1:e} | Time : {2:.2f} s".format(e+1,loss_per_epoch,time.time()-start))
			self.error_per_epoch.append(loss_per_epoch)

	def show_loss_plot(self,detailed=True):
		x = self.errors if detailed else self.error_per_epoch
		plt.plot(x)
		plt.show()

	def predict(self,x):
		res = self.forward_propagate(x).flatten()
		res[np.abs(res-1) < np.abs(res-0)] = 1
		res[res != 1] = 0
		return res.astype(int)


if __name__ == '__main__':
	a = ANNClassifier(random_state=0)
	a.add(Layer(3))
	a.add(Layer(2,activation_function='softplus'))
	a.add(Layer(3,activation_function='softplus'))
	a.add(Layer(1))
	input_val = np.array([[1,2,3],[2,4,6]])
	expected = np.array([[1],[100]])
	a.SGD(input_val,expected,0.5)