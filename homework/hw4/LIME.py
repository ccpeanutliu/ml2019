
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
import lime
from lime import lime_image
from skimage.segmentation import slic,mark_boundaries


# In[2]:


model = load_model('hw3_model.h5')


x_test = np.load("x_test.npy")


img = x_test[0]
img = img.reshape(1,48,48,1)
pred = model.predict(img).reshape(7)


# In[5]:



#plt.imshow(img_rgb)
#print(pred.shape,y_test.shape,img_rgb.shape)


# In[6]:



# load data and model
#x_train = torch.load('data/train_data.pth')
#x_label = torch.load('data/train_label.pth')

# Lime needs RGB images
# TODO:
# x_train_rgb = ?

# two functions that lime image explainer requires


def predict(input):
    input = input.transpose(3,1,2,0)
    use = input[0].transpose(2,0,1)
    use = use.reshape(10,48,48,1)
    pre = model.predict(use)
    return(pre)
    # Input: image tensor
    # Returns a predict function which returns the probabilities of labels ((7,) numpy array)
    # ex: return model(data).numpy()
    # TODO:
    # return ?


# In[7]:


def segmentation(input):
    segments = slic(input, n_segments=40, compactness=10)
    return(segments)

    # Input: image numpy array
    # Returns a segmentation function which returns the segmentation labels array ((48,48) numpy array)
    # ex: return skimage.segmentation.slic()
    # TODO:
    # return ?


# In[8]:


# Initiate explainer instance
for i in range(7):
    img_rgb = np.array([x_test[i],x_test[i],x_test[i]])
    img_rgb = img_rgb.transpose((3,1,2,0))
    img_rgb = img_rgb.reshape(48,48,3)

    explainer = lime_image.LimeImageExplainer()

    # Get the explaination of an image
    def explain(instance, predict_fn, **kwargs):
        np.random.seed(16)
        return exp.explain_instance(instance, predict_fn, **kwargs)
    
    explaination = explainer.explain_instance(
                                image= img_rgb, #x_train_rgb[idx], 
                                classifier_fn=predict,
                                labels = [0,1,2,3,4,5,6],
                                segmentation_fn=segmentation
                            )

    # Get processed image
    image, mask = explaination.get_image_and_mask(
                                    label=i,
                                    positive_only=False,
                                    hide_rest=False,
                                    num_features=5,
                                    min_weight=0.0
                                )
    # save the image
    plt.imsave(sys.argv[1] +'fig3_'+str(i)+'.jpg',image)

