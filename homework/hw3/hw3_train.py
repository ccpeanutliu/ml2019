
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
from keras.layers.advanced_activations import LeakyReLU


# In[2]:


train_y = []
train_x = []

for i in range(28709):
    train_y.append([])
    train_x.append([])


# In[3]:


count = 0
with open(sys.argv[1], newline='') as csvfile:
    rows = csv.reader(csvfile)
    for row in rows:
        if(row[0] != "label"):
            train_y[count].append(row[0])
            train_x[count].append(row[1])
            count += 1
        


# In[ ]:


x = []
for i in range(len(train_x)):
    x.append(train_x[i][0].split(' '))


# In[ ]:


for i in range(len(train_x)):
    for j in range(len(x[0])):
        x[i][j] = float(x[i][j])


# In[ ]:


x = np.matrix(x)
temp_x = x


# In[ ]:


y_use = np.array(train_y)


# In[ ]:


x = []
data = 0
for i in range(28709):
    x.append([])
    for j in range(48):
        x[i].append([])    


# In[ ]:


for i in range(temp_x.shape[0]):
    for j in range(temp_x.shape[1]):
        if j != 0 and j % 48 == 0:
            data += 1
        x[i][data].append(temp_x[i,j])
        
    data = 0


# In[ ]:


x = np.array(x)


# In[ ]:


y_use = np_utils.to_categorical(y_use)


# In[ ]:


x_reshape = x.reshape(28709,48,48,1).astype('float32') 


# In[ ]:


x_use = x_reshape/225



# In[16]:


x1_use = x_use[0:26000]
x2_use = x_use[26001:28709]
y1_use = y_use[0:26000]
y2_use = y_use[26001:28709]

# In[18]:


model=Sequential()
model.add(Conv2D(filters=64,kernel_size=(3,3),padding='same',input_shape=(48,48,1)))
model.add(LeakyReLU(alpha=0.03))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.3))


# In[19]:


for i in range(1):
    model.add(Conv2D(filters=128,kernel_size=(3,3),padding='same'))
model.add(LeakyReLU(alpha=0.03))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.3))


# In[20]:


for i in range(1):
    model.add(Conv2D(filters=256,kernel_size=(3,3),padding='same'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(LeakyReLU(alpha=0.03))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.3))


# In[21]:


for i in range(1):
    model.add(Conv2D(filters=512,kernel_size=(3,3),padding='same'))
model.add(LeakyReLU(alpha=0.03))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.3))


# In[22]:


model.add(Flatten())
model.add(Dense(256,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(512,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Dense(7,activation='softmax'))


# In[23]:


model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])


# In[24]:


from keras.callbacks import ReduceLROnPlateau


# In[25]:


from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=25,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=[0.8, 1.2],
    horizontal_flip=True
    )


datagen.fit(x1_use)
train_generator = datagen.flow(x1_use,y1_use,batch_size = 200)
#learning_rate_function = ReduceLROnPlateau(monitor='val_acc',patience=2,epsilon=0.00001,verbose=1,factor=0.2)
train_history2 = model.fit_generator(train_generator, steps_per_epoch=1500,epochs=40,verbose=1,validation_data=(x2_use,y2_use))#,callbacks=[learning_rate_function])


# In[26]:


model.save('hw3_train_model.h5')

