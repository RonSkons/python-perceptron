import numpy as np
import cv2
import os
import re
import random

bias = 0
width, height = 800, 800
dimensions = (width, height)
zero_meaning = "capybara"
one_meaning = "star"
training_dir = "training/capystar/"
identification_dir = "testing/capystar/"

try:
	weights = np.loadtxt("weights.txt")
except OSError:
	print("Could not find saved weights, creating weights.txt.")
	weights = np.zeros((height, width))
	np.savetxt("weights.txt", weights)

if weights.shape != (height, width):
	print("Saved weights have dimensions %i x %i, while provided dimensions are %i x %i." % (weights.shape[1], weights.shape[0], width, height))
	print("(O)verwrite the saved weights with a correctly-sized empty array, (s)cale the saved weights to the provided dimensions, or (r)eplace width and height parameters with the dimensions of the saved weights?")
	x = input()
	
	if x.lower()[0] == "o":
		print("Overwriting weights.txt")
		weights = np.zeros((height, width))
		np.savetxt("weights.txt", weights)
	elif x.lower()[0] == "s":
		print("Scaling weights.txt")
		weights = cv2.resize(weights, (height, width))
		np.savetxt("weights.txt", weights)
	else:
		width = weights.shape[1]
		height = weights.shape[0]
		print("New image dimensions: %i x %i" % dimensions)

# weighted sum of arr, with given weights. both are numpy arrays.
def weighted_sum(arr, wghts):
	weighted = arr*wghts
	return np.sum(weighted)

# categorizes scores as 0 or 1 by comparing them against the bias
def step_function(score):
	if score > bias:
		return 1
	else:
		return 0

# processes image at given path
# returns numpy array of pixel values ranging from 0 (white) to 1 (black)
def img_to_array(path):

	img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA) 
	
	# replace transparent with white 
	trans_mask = img[:,:,3] == 0
	img[trans_mask] = [255, 255, 255, 255]
	
	# greyscale, set black to 1 and white to 0
	new_img = cv2.bitwise_not(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))/255
	
	if new_img.shape != dimensions:
		# image is incorrectly sized; resize
		new_img = cv2.resize(new_img, dimensions)

	return new_img

# trains the network with image at path
# truth represents the categorization of the image—either 0 or 1
# returns updated weights
def train_single(path, truth, weights):
	img = img_to_array(path)
	attempt = step_function(weighted_sum(img, weights))
	if (truth == 1) and (attempt == 0):
		print("False negative, correcting")
		return weights + img
	elif (truth == 0) and (attempt == 1):
		print("False positive, correcting")
		return weights - img
	else:
		print("Correct guess")
		return weights

# train the network with all appropriately-named images in dir
# returns updated weights
def train_all(dir, weights):
	files = list(filter(re.compile('[01].*\.(png|jpg|jpeg)').match, os.listdir(dir)))
	print("Found ", len(files)," images. Training.")
	random.shuffle(files)
	
	last_weights = weights
	for file in files:
		last_weights = train_single(dir+file, int(file[0]), last_weights)

	print("Done training.")
	return last_weights

def identify_single(path, weights):
	img = img_to_array(path)
	sum = weighted_sum(img, weights)
	#print(sum+bias)
	result = step_function(sum)
	return zero_meaning if (result == 0) else one_meaning

def identify_all(dir, weights):
	files = list(filter(re.compile('.*\.(png|jpg|jpeg)').match, os.listdir(dir)))
	print("Found", len(files),"images. Identifying.")
	
	for file in files:
		print(file, "is", identify_single(dir+file, weights))

	print("Done Identifying.")

def dump_weights(weights):
	cv2.imwrite("debug.png", (255*(weights - np.min(weights))/np.ptp(weights)).astype(int))

x = ""
while x != "q":
	print("(T)rain, (i)dentify, or (r)eset saved weights, (d)ebug, or (q)uit.")
	x = input().lower()[0]
	if x == "t":
		weights = train_all(training_dir, weights)
		np.savetxt("weights.txt", weights)
	elif x == "r":
		weights = np.zeros((height, width))
		np.savetxt("weights.txt", weights)
		print("Reset weights.")
	elif x == "i":
		identify_all(identification_dir, weights)
	elif x == "d":
		dump_weights(weights)
		print("Dumped weights to image.")