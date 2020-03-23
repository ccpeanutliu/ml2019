
# coding: utf-8

# In[1]:


import sys
import csv
import numpy as np
from keras.utils import np_utils
from keras.datasets import mnist 
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.regularizers import l2
from keras.layers.normalization import BatchNormalization
import keras.backend as K
import matplotlib.pyplot as plt


# In[2]:


model = load_model("hw3_model.h5")


# data start
train_y = []
train_x = []

for i in range(28709):
    train_y.append([])
    train_x.append([])

count = 0
with open(sys.argv[1], newline='') as csvfile:
    rows = csv.reader(csvfile)
    for row in rows:
        if(row[0] != "label"):
            train_y[count].append(row[0])
            train_x[count].append(row[1])
            count += 1

x = []
for i in range(len(train_x)):
    x.append(train_x[i][0].split(' '))



for i in range(len(train_x)):
    for j in range(len(x[0])):
        x[i][j] = float(x[i][j])



x = np.matrix(x)
temp_x = x



y_use = np.array(train_y)



x = []
data = 0
for i in range(28709):
    x.append([])
    for j in range(48):
        x[i].append([])    


for i in range(temp_x.shape[0]):
    for j in range(temp_x.shape[1]):
        if j != 0 and j % 48 == 0:
            data += 1
        x[i][data].append(temp_x[i,j])
        
    data = 0



x = np.array(x)

y_use = np_utils.to_categorical(y_use)



x_reshape = x.reshape(28709,48,48,1)

x_test = np.array([x_reshape[28647],x_reshape[28650],x_reshape[28648],x_reshape[28649],x_reshape[28638],x_reshape[28651],x_reshape[28652]]).astype("float32")
y_test = np.array([0,1,2,3,4,5,6])

x_test = x_test/255

#data done

np.save("x_test.npy",x_test)
# In[4]:



x_test_use = np.load("x_test.npy")


# In[5]:


def compile_saliency_function(model):
    """
    Compiles a function to compute the saliency maps and predicted classes
    for a given minibatch of input images.
    """
    inp = model.layers[0].input
    outp = model.layers[-1].output
    max_outp = K.max(outp, axis=1)
    saliency = K.gradients(K.sum(max_outp), inp)[0]
    max_class = K.argmax(outp, axis=1)
    return K.function([inp], [saliency, max_class])

sal = compile_saliency_function(model)([x_test_use, 0])


# In[6]:


sal[0] = sal[0].reshape(7,48,48)


# In[7]:


salimg = sal[0][0:7]


# In[8]:


salimg = np.array(salimg)


# In[9]:


salimg.shape


# In[15]:


for i in range(7):
    img = plt.imshow(salimg[i])
    plt.title(str(i))
    img.set_cmap('gray')
    tempstr = sys.argv[2] + 'fig1_'+str(i)+'.jpg'
    print(tempstr)
    plt.savefig(tempstr)

