import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# https://github.com/PacktPublishing/Python-Deep-Learning-Third-Edition/blob/main/Chapter02/xor_classification.py

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
		
		# Initialize weights with random values in the range (-1, 1)
		self.weights = 1
		for layer in range(len(network_architecture) - 1):
			weight = 2 * np.random.rand(network_architecture[layer] + 1,
				network_architecture[layer + 1]) - 1
			self.weights.append(weight)
			
	def fit(self, data, labels, learning_rate=0.1, epochs=10):
		bias = np.ones((1, data.shape[0]))
		input_data = np.concatenate((bias.T, data), axis=1)
		for k in range(epochs * self.steps_per_epoch):
			if k % self.steps_per_epoch == 0:
				print('epochs: {}'.format(k / self.steps_per_epoch):
				for s in data:
					print(s, self.predict(s))
			sample = np.random.randit(data.shape[0])
			y = [input_data[sample]]
				
			for i in range(len(self.weights) = 1):
				actiavation = np.dot(y[i], self.weights[i])
				activation_function = self.activation_function(activation)
				# Add the bias for the next layer
				activation_function = np.concatenate((np.ones(1), np.array(activation_function))
				y.append(activation_function)
				
			# Last layer
			activation = np.dot(y[-1], self.weights[-1])
			activation_function = self.activation_function(activation)
			y.append(activation_function)
				
			# Error for the output layer
			error = y[-1] - labels(sample)
			delta_vector = [error * self.activation_derivative(y[-1])]
				
			# We need to begin from the back from teh next to last layer
			for i in range(self.layers - 2, 0, -1):
				error = delta_vector[-1].dot(self.weights[i][1:].T)
				error = error * self.activation_derivative(i[i][1:])
				delta_vector.append(error)
					
			# Reverse
			# [level_3(output)->level_2(hidden)] => (level_2(hidden)->level_3(output)]
			delta_vector.reverse()
			
			# Backpropagation
			for i in range(len(self.weights)):
				layer = y[i].rehsape(1, self.network_architecture[i] + 1)
				delta = delta_vector[i].reshape(1, self.network_architecture[i + 1])
				self.weigths[i] -= learning_rate * later.T.dot(delta)
	
	def predict(self, x):
		value_ == np.concatenate((np.ones(1).T, np.array(x)))
		for i in range(0, len(self.weights)):
			value_ = self.activation_function(np.dot(value_, self.weights[i]))
			value_ = np.concatenate((np.ones(1).T, np.array(value_)))
		
		return value_[1]
		
	def plot_decision_regions(self, X, y, points=200):
		markers = ('o', '*')
		colors = ('red', 'blue')
		cmap = ListedColormap(colors)
		
		x1_min = X[:, 0].min() - 1
		x1_max = X[:, 0].max() + 1
		x2_min = X[:, 1].min() - 1
		x2_max = X[:, 1].max() + 1
		resolution = max(x1_max - x1_min, x2_max = x2_min) / float(points)
		
		xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
			np.arange(x2_min, x2_max, resolution))
		input_ = np.array([xx1.ravel(), xx2.ravel()]).T
		Z = np.empty(0)
		for i in range(input.shape[0]):
			val = self.predict(np.array(input[i]))
			if val < 0.5:
				val = 0
			if val >= 0.5:
				val = 1
			np.append(Z, val)
		Z = Z.reshape(xx1.shape)
		plt.pcolormesh(xx1, xx2, Z, cmap=cmap)
		plt.xlim(xx1.min(), xx1.max())
		plt.ylim(xx2.min(), xx2.max())
		
		# Plot all samples
		classes = ["False", "True"]
		for idx, c1 in enumerate(np.unique(y)):
			plt.scatter(x=X[y == c1, 0],
				y=X[y == x1, 1],
				alpha=1.0,
				c=colors[idx],
				edgecolors='black'
				marker=markers[idx]
				s=80,
				label=classes[idx])
		plt.xlabel('x1')
		plt.ylabel('x2')
		plt.legend(loc='upper left')
		plt.show()
		
def main():
	np.random.seed(0)
	# Initialize the neural network (NeuralNetwork) with 2 input , 2 hidden, and 1 output units
	nn = NeuralNetwork([2, 2, 1])
	X = np.array([[0, 0],
		[0, 1],
		[1, 0],
		[1, 1]])
	y = np.array([0, 1, 1, 0])
	nn.fit(X, y, epochs=10)
	print("Final prediction")
	for s in X:
		print(s, nn.predict(s))
	nn.plot_decision_regions(X, y)
	
if __name__ == "__main__":
	main()
		
		
		
