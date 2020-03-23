import sys, os
import csv
import numpy as np
from keras.models import load_model
import keras.backend as K
import matplotlib.pyplot as plt
import time


model = load_model('hw3_model.h5')
img_width = 48
img_height = 48


start = time.time()
fg = plt.figure()	
plt.axis('off')
plt.title("Filters of layer conv2d_1")


for filter_index in range(16):

	print(filter_index)
	
	def compile_gradient_function(model):
		input_img = model.input
		layer_output = model.layers[2].output
		loss = K.mean(layer_output[:, :, :, filter_index])
		grads = K.gradients(loss, input_img)[0]
		grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
		return K.function([input_img], [grads])

	# we start from a gray image with some noise
	np.random.seed(5);
	input_img_data = np.random.random((1,48,48,1))
	# run gradient ascent for 20 steps
	step = 0.01

	for i in range(20):

	    grads_value = compile_gradient_function(model)([input_img_data])
	    input_img_data += grads_value[0] * step


	print(input_img_data.shape)
	input_img_data = np.reshape(input_img_data, (48, 48))

	
	fg.add_subplot(4,4,filter_index+1)
	plt.imshow(input_img_data, cmap='spring')
	plt.axis('off')

	#plt.colorbar()
	#name = './filter_index/filter'+str(filter_index)+'.png'
	#print(name)
plt.axis('off')
plt.savefig(sys.argv[1] + "fig2_1.jpg")
plt.axis('off')
end = time.time()
print(end-start)
#plt.clf()
