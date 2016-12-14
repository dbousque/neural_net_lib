

import math

def sigmoid(res):
	lol = 0.0
	try:
		lol = math.exp(-res)
	except:
		if res >= 0.0:
			lol = 0.0
		else:
			lol = 1000000000.0

	return (1 / (1 + lol))

class Layer:

	def __init__(self, neurons):
		self.nb_neurons = len(neurons)
		self.neurons = neurons

class NeuralNet:

	def __init__(self, filename):
		fich = open(filename).read()
		lines = fich.split("\n")
		layers = []
		tmp_lines = []
		for i,line in enumerate(lines):
			if len(line) == 0:
				layers.append(tmp_lines)
				if len(lines[i + 1]) == 0:
					break
				tmp_lines = []
			else:
				tmp_lines.append(line)
		for i,lay in enumerate(layers):
			for x,neuron in enumerate(lay):
				layers[i][x] = [float(elt) for elt in neuron.split(";")[:-1]]
		self.nb_layers = len(layers)
		self.layers = []
		for elt in layers:
			self.layers.append(Layer(elt))

	def predict(self, input, ind):
		tmp_out = []
		print len(input)
		print self.layers[ind].nb_neurons
		print len(self.layers[ind].neurons[0])
		for i in range(self.layers[ind].nb_neurons):
			res = 0.0
			for x in range(len(input)):
				res += input[x] * self.layers[ind].neurons[i][x]
			tmp_out.append(sigmoid(res))
		if ind == self.nb_layers - 1:
			return tmp_out
		tmp_out.append(1.0)
		return self.predict(tmp_out, ind + 1)

	def generate_image(self, output):


neural_net = NeuralNet("nn1.nn")
"""while True:
	image = raw_input()
	bytes = []
	with open(image[1:-2], "rb") as f:
		byte = f.read(1)
		i = 0
		while byte != "":
			if i < 122:
				print "\theader[" + str(i) + "] = " + str(ord(byte)) + ";"
			bytes.append(ord(byte) / 255.0)
			byte = f.read(1)
			i += 1
		bytes = bytes[122:]
		bytes.append(1.0)
		res = neural_net.predict(bytes, 0)
		print "probabilites : ", res
		print "recognized   : ", res.index(max(res))"""