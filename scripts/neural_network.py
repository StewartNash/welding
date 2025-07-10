import numpy as np

def tanh(x):
	return (1.0 - np.exp(-2 * x)) / (1.0 + np.exp(-2 * x))

def tanh_derivative(x):
	return (1 + tanh(x)) * (1 - tanh(x))

class NeuralNetwork:
	def __init__(self, network_architecture):
		self.activation_function = tanh
		self.activation_derivative = tanh_derivative
		self.layers = len(network_architecture)
		self.steps_per_epoch = 1000
		self.network_architecture = network_architecture
