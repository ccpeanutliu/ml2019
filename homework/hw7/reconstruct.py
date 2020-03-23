#!/usr/bin/env python
# coding: utf-8

# code resource: https://hackmd.io/nBZQRocKSe204LRdlO5XUQ
# ppt resource: https://docs.google.com/presentation/d/1A_o2m6_bMFtOur660ZpBT162kB5B8DI_iHQaWYUMMQc/edit#slide=id.g58c65b53a1_0_116

# In[1]:
import os
import sys
import numpy as np 
from skimage.io import imread, imsave

# In[2]

IMAGE_PATH = sys.argv[1]

test_image = [sys.argv[2]]

k = 1

# In[3]
def process(M): 
    M -= np.min(M)
    M /= np.max(M)
    M = (M * 255).astype(np.uint8)
    return M

# In[4]
# Record the shape of images 

img_shape = (600,600,3)
img_data = []
for i in range(415):
    tmp = imread('./' + IMAGE_PATH + str(i) + '.jpg')  
    img_data.append(tmp.flatten())

# In[5]
training_data = np.array(img_data).astype('float32')

# Calculate mean & Normalize
mean = np.mean(training_data, axis = 0)  
training_data -= mean 

# Use SVD to find the eigenvectors 
u, s, v = np.linalg.svd(training_data, full_matrices = False)

# In[6]

# Pro 1.c
for x in test_image: 
    # Load image & Normalize
    picked_img = imread('./' + IMAGE_PATH + '/' + str(x))   
    X = picked_img.flatten().astype('float32') 
    X -= mean
    
    # Compression
    weight = np.array([X.dot(np.transpose(v[i])) for i in range(k)])  
    
    # Reconstruction
    reconstruct = process(weight.dot(v[:k]) + mean)
    imsave(sys.argv[3], reconstruct.reshape(img_shape)) 