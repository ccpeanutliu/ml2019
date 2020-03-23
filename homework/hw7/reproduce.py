#!/usr/bin/env python
# coding: utf-8

# In[1]:


import csv
import numpy as np
from skimage import io, color
import sys
from sklearn.cluster import KMeans
from keras.models import load_model
import pickle


# In[2]:


with open('Strong_KMeans.pickle', 'rb') as f:
    KMeans = pickle.load(f)
with open('sklearn_pca.pickle', 'rb') as f:
    sklearn_pca = pickle.load(f)
model = load_model("autoencoder.h5")

# In[3]:


img_n = 40000
img = [] 
for i in range(1,img_n+1):
    if i < 10:
        img_str = "00000"
    elif i < 100:
        img_str = "0000"
    elif i < 1000:
        img_str = "000"
    elif i < 10000:
        img_str = "00"
    else:
        img_str = "0"
    img.append(io.imread(sys.argv[1] + img_str + str(i) + ".jpg"))
img = np.array(img)

# In[4]:


x_train = img/255
x_val = img[36000:40000]/255
x_flat = x_train.reshape(40000,32*32*3)
img_auto = model.predict(x_train)
np.random.seed(1200)#1230
Y_sklearn = sklearn_pca.fit_transform(x_flat)


# In[5]:


X_clustered = KMeans.fit_predict(Y_sklearn)


# In[6]:


labels = X_clustered
n_test = 1000000
x_label = np.zeros([n_test,2])
count = 0
with open(sys.argv[2], newline='') as csvfile:
    rows = csv.reader(csvfile)
    for row in rows:
        if(row[0] != "id"):
            x_label[count,0] = row[1]
            x_label[count,1] = row[2]
            count += 1
x_label = np.uint(x_label)


# In[7]:


test_labels = []
for i in range(2):
    test_labels.append([])
for i in range(n_test):
    tmp = [int(x_label[i,0]-1), int(x_label[i,1]-1)]
    test_labels[0].append(labels[tmp[0]])
    test_labels[1].append(labels[tmp[1]])
test_labels = np.array(test_labels)


# In[8]:


output = np.zeros([n_test])
for i in range(n_test):
    if test_labels[0,i] == test_labels[1,i]:
        output[i] = 1
    else:
        output[i] = 0


# In[9]:


sum(output)#501329.0


# In[10]:


with open(sys.argv[3], 'w', newline='') as csvfile:
  csvfile.write('id,label\n')
  for i, v in enumerate(output):
    csvfile.write('%d,%d\n' %(i, output[i]))


# In[ ]:




