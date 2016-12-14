

from mnist import get_images, save_image_to_file
import random, math

def sigmoid(value):
	try:
		return (1.0 / (1 + math.exp(-value)))
	except OverflowError:
		if (value > 0):
			return (1.0)
		return (0.0)

def get_cost(neural_net_output, expected_output):
	val = [(neural_net_output[i] - expected_output[i]) ** 2 for i in range(len(expected_output))]
	return sum(val) / len(expected_output)

class Neuron:

	def __init__(self, nb_inputs):
		self.weights = [random.uniform(-10.0, 10.0) for i in range(nb_inputs)]
		self.value = 0.0

	def calculate_value(self, inp, activation_func):
		self.value = -1 * self.weights[0]
		for i,neuron in enumerate(inp.neurons):
			self.value += neuron.value * self.weights[i + 1]
		self.value = activation_func(self.value)

class Layer:

	def __init__(self, nb_neurons, nb_inputs):
		self.neurons = [Neuron(nb_inputs) for i in range(nb_neurons)]

class NeuralNet:

	def __init__(self, neurons_per_layer, learning_rate=0.001):
		self.layers = [Layer(neurons_per_layer[i], 0) if i == 0 else Layer(neurons_per_layer[i], neurons_per_layer[i - 1] + 1) for i in range(len(neurons_per_layer))]
		self.learning_rate = learning_rate

	def calculate_layer(self, layer_nb, activation_func):
		for neuron in self.layers[layer_nb].neurons:
			neuron.calculate_value(self.layers[layer_nb - 1], activation_func)

	def get_output(self, activation_func):
		for i in range(len(self.layers) - 1):
			self.calculate_layer(i + 1, activation_func)
		return [out_neur.value for out_neur in self.layers[-1].neurons]

	def gradient_descent(self, inputs, expected_outputs, cost_func=get_cost, activation_func=sigmoid):
		actual_gradient_descent(self, inputs, expected_outputs, cost_func, activation_func)

	def set_first_layer(self, inp):
		for i,elt in enumerate(inp):
			self.layers[0].neurons[i].value = elt

	def predict(self, inp, activation_func=sigmoid):
		self.set_first_layer(inp)
		return self.get_output(activation_func)

	def update_weights(self, gradient):
		for x,layer in enumerate(self.layers):
			for y,neuron in enumerate(layer.neurons):
				for z,weight in enumerate(neuron.weights):
					neuron.weights[z] += -gradient[x][y][z] * self.learning_rate

def get_partial_der(neural_net, layer_nb, neuron_nb, weight_nb, cost_func, activation_func, expected_output):
	#print "CALLED"
	ori_cost = cost_func(neural_net.get_output(activation_func), expected_output)
	ori_val = neural_net.layers[layer_nb].neurons[neuron_nb].weights[weight_nb]
	neural_net.layers[layer_nb].neurons[neuron_nb].weights[weight_nb] += neural_net.learning_rate
	new_cost = cost_func(neural_net.get_output(activation_func), expected_output)
	neural_net.layers[layer_nb].neurons[neuron_nb].weights[weight_nb] = ori_val
	#print "RETURN"
	return ((new_cost - ori_cost) / neural_net.learning_rate)

def get_gradient(neural_net, cost_function, activation_func, expected_output):
	partial_derivatives = []
	for x,layer in enumerate(neural_net.layers):
		partial_derivatives.append([])
		for y,neuron in enumerate(layer.neurons):
			print "new neuron " + str(x) + " " + str(y)
			partial_derivatives[-1].append([])
			for z,weight in enumerate(neuron.weights):
				#print "new weight"
				partial_derivatives[-1][-1].append(get_partial_der(neural_net, x, y, z, cost_function, activation_func, expected_output))
	print "return partial_derivatives"
	return (partial_derivatives)

def actual_gradient_descent(neural_net, inputs, expected_outputs, cost_func, activation_func):
	for i,inp in enumerate(inputs):
		print "gradient : " + str(i)
		neural_net.set_first_layer(inp)
		gradient = get_gradient(neural_net, cost_func, activation_func, expected_outputs[i])
		neural_net.update_weights(gradient)

def score(neural_net, inputs, expected_outputs):
	correct = 0
	incorrect = 0
	for i,inp in enumerate(inputs):
		if i % 500 == 0:
			print i
		out = neural_net.predict(inp)
		if out.index(max(out)) == expected_outputs[i].index(max(expected_outputs[i])):
			correct += 1
		else:
			incorrect += 1
	print "got " + str(correct) + " correct."
	print "got " + str(incorrect) + " incorrect."

LEARNING_RATE = 0.001
COST_FUNC = get_cost
ACTIVATION_FUNC = sigmoid

images = get_images()
print "got images.\n"
#save_image_to_file(images[0].pixels, "test1.tif")
inputs = [image.pixels for image in images]
expected_outputs = [[0] * 10 for image in images]
training_inp = inputs[:5000]
training_out = expected_outputs[:5000]
for i,elt in enumerate(images):
	expected_outputs[i][elt.label] = 1
neural_net = NeuralNet([28 * 28, 100, 10])
#score(neural_net, training_inp, training_out)
neural_net.gradient_descent(training_inp, training_out)
score(neural_net, training_inp, training_out)