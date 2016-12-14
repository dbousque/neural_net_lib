from PIL import Image as Img
from random import randint

class Image:

	def __init__(self, pixels, label):
		self.pixels = pixels
		self.label = label

	def __str__(self):
		return "nb pixels : " + str(len(self.pixels)) + " ; label : " + str(self.label)

def get_images_n_labels():
	bytes = []
	f = open("train-images.idx3-ubyte", "rb")
	try:
		byte = f.read(1)
		while byte != "":
			# Do stuff with byte.
			bytes.append(ord(byte))
			byte = f.read(1)
	finally:
		f.close()

	pixels = bytes[16:]
	images = []
	i = 0
	while i <= len(pixels):
		if i % (28 * 28) == 0 and i != 0:
			images.append(pixels[i - (28 * 28):i])
		i += 1
	labels = []
	f = open("train-labels.idx1-ubyte", "rb")
	try:
		byte = f.read(1)
		while byte != "":
			# Do stuff with byte.
			labels.append(ord(byte))
			byte = f.read(1)
	finally:
		f.close()
	labels = labels[8:]
	return images, labels

def save_image_to_file(image, filename):
	new_img = Img.new("L", (28, 28), "white")
	new_img.putdata(image)
	new_img.save(filename)

def get_images():
	images, labels = get_images_n_labels()
	return [Image(images[i], labels[i]) for i in range(len(images))]
	
#print labels[10]
#save_image_to_file(images[10], "image10.bmp")
#print len(images), len(labels)
images = get_images()
for i in range(200):
	save_image_to_file(images[i].pixels, "images/image_" + str(i) + ".tif")