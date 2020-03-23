
# coding: utf-8

# In[5]:


import csv
import math
from PIL import Image
import matplotlib.pyplot as plt
import keras
from keras.applications import vgg19
from keras.models import load_model
from keras.utils import np_utils
from keras.datasets import mnist 
from keras.models import Sequential
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import keras.backend as K
import sys

# In[6]:


K.tensorflow_backend._get_available_gpus()


# In[7]:


model = keras.applications.vgg19.VGG19(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)


# In[8]:


img_arr = []
initial_class_arr = []
origin_img = []
for i in range(200):
    if(i < 10):
        read = sys.argv[1] + "/00" + str(i) + ".png"
    elif(i >= 10 and i < 100):
        read = sys.argv[1] + "/0" + str(i) + ".png"
    else:
        read = sys.argv[1] + "/" + str(i) + ".png"
    I = Image.open(read)
    I = np.array(I)
    origin_img.append(I)
    I = vgg19.preprocess_input(I)
    img_arr.append(I)
    preds = model.predict(I.reshape(1,224,224,3))
    initial_class = np.argmax(preds)
    initial_class_arr.append(initial_class)
    
img_arr = np.array(img_arr)


# In[9]:


def compile_grad_function(model,initial_class):
    target = K.one_hot(initial_class, 1000)
    inp = model.input
    outp = model.output
    #print(inp)
    loss = K.categorical_crossentropy(target, model.output)

    #print(outp)
    grad = K.gradients(loss, inp)[0]
    
    #print(grad)
    return K.function([inp], [grad])


# In[11]:


adver = []
step = 4
epoch = 200
for i in range(epoch):
    
    K.clear_session()
    model = vgg19.VGG19(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
    sess = K.get_session()
    
    if(i % 20 == 0):
        print(i)
    input_img = np.copy(img_arr[i].reshape(1,224,224,3).astype('float32'))
    grads_value = compile_grad_function(model,initial_class_arr[i])([input_img,0])
    delta = K.sign(grads_value[0])
    
    input_img += delta * step
    input_img = sess.run(input_img, feed_dict={model.input:img_arr[i].reshape(1,224,224,3)})
    input_img = input_img.reshape(224,224,3).astype('int')
    
    adver.append(input_img)
adver = np.array(adver)


# In[12]:


output_img = np.copy(adver+[103.939, 116.779, 123.68])


# In[13]:


output_img[output_img > 255] = 255
output_img[output_img < 0] = 0


# In[14]:


output = np.zeros_like(output_img)
output[:,:,:,0] = output_img[:,:,:,2]
output[:,:,:,1] = output_img[:,:,:,1]
output[:,:,:,2] = output_img[:,:,:,0]


# In[15]:


output_arr = np.copy(output/255)


# In[16]:


number = []
for i in range(10):
    number.append("00"+str(i))
for i in range(10, 100):
    number.append("0"+str(i))
for i in range(100, 200):
    number.append(str(i))


for i in range(200):
    name = sys.argv[2] + "/" + number[i] + ".png"
    plt.imsave(name, output_arr[i])

