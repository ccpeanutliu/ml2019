
# coding: utf-8

# In[32]:


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
from keras.layers.advanced_activations import LeakyReLU

test_x = []

for i in range(7178):
    test_x.append([])

count_t = 0
with open(sys.argv[1], newline='') as csvfile:
    rows = csv.reader(csvfile)
    for row in rows:
        if(row[0] != "id"):
            test_x[count_t].append(row[1])
            count_t += 1

x_test = []
for i in range(len(test_x)):
    x_test.append(test_x[i][0].split(' '))

for i in range(len(test_x)):
    for j in range(len(x_test[0])):
        x_test[i][j] = float(x_test[i][j])

x_test = np.matrix(x_test)
temp_x_test = x_test

x_test = []
data_test = 0
for i in range(7178):
    x_test.append([])
    for j in range(48):
        x_test[i].append([])


for i in range(temp_x_test.shape[0]):
    for j in range(temp_x_test.shape[1]):
        if j != 0 and j % 48 == 0:
            data_test += 1
        x_test[i][data_test].append(temp_x_test[i,j])
        
    data_test = 0



x_test = np.array(x_test)


x_test_reshape = x_test.reshape(7178,48,48,1).astype('float32') 
x_test_use = x_test_reshape/225


model = load_model("hw3_model.h5")



# In[35]:


prediction = model.predict_classes(x_test_use)


# In[38]:


with open(sys.argv[2], 'w', newline='') as csvfile:
  csvfile.write('id,label\n')
  for i, v in enumerate(prediction):
    csvfile.write('%d,%d\n' %(i, prediction[i]))

