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

x_test_use = np.load("x_test.npy")
img_test = x_test_use[0].reshape(1,48,48,1)
temp = img_test
start = time.time()
fg = plt.figure()	
plt.axis('off')
plt.title("Filters of layer conv2d_1")

def compile_gradient_function(model):
	input_img = model.input
	layer_output = model.layers[2].output
	loss = K.mean(layer_output[:, :, :, 0])
	grads = K.gradients(loss, input_img)[0]
	grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
	return K.function([input_img], [grads,layer_output])

grads_value = compile_gradient_function(model)([img_test])
gg = grads_value[1].reshape(48,48,64)
gg = gg.transpose((2,0,1))

for filter_index in range(64):
	img_test = gg[filter_index]
	#print(img_test.shape)
	img_test = np.reshape(img_test, (48, 48))

	
	fg.add_subplot(8,8,filter_index+1)
	plt.imshow(img_test, cmap='hot')
	plt.axis('off')

	#plt.colorbar()
	#name = './filter_index/filter'+str(filter_index)+'.png'
	#print(name)
plt.axis('off')
plt.savefig(sys.argv[1] + "fig2_2.jpg")
plt.axis('off')
end = time.time()
print(end-start)
#plt.clf()
