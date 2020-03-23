
# coding: utf-8

# In[1]:

from keras.models import Model, load_model
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dropout
from keras.layers import Activation, BatchNormalization, Add, Reshape, DepthwiseConv2D
from keras.utils.vis_utils import plot_model
from keras.layers.advanced_activations import LeakyReLU
from keras import backend as K
import numpy as np
from keras.utils import np_utils
import sys


# In[22]:

n = 28709
train_y = []
train_x = []
for i in range(n):
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
x = np.array(x)
y = np.array(train_y)


# In[23]:

x = x.reshape(n,48,48,1)
y = y.reshape(n)


# In[24]:

y_cat = np_utils.to_categorical(y)


# In[ ]:

np.save("img_train.npy",x)
np.save("img_label.npy",y_cat)


# In[2]:

np.random.seed(1230)


# In[3]:


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def relu6(x):
    """Relu 6
    """
    return K.relu(x, max_value=6.0)


def _conv_block(inputs, filters, kernel, strides):
    """Convolution Block
    This function defines a 2D convolution operation with BN and relu6.
    # Arguments
        inputs: Tensor, input tensor of conv layer.
        filters: Integer, the dimensionality of the output space.
        kernel: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
        strides: An integer or tuple/list of 2 integers,
            specifying the strides of the convolution along the width and height.
            Can be a single integer to specify the same value for
            all spatial dimensions.
    # Returns
        Output tensor.
    """

    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    x = Conv2D(filters, kernel, padding='same', strides=strides)(inputs)

    x = BatchNormalization(axis=channel_axis)(x)
    #x = BatchNormalizationF16(axis=channel_axis)(x)
    return LeakyReLU(alpha=0.03)(x)


def _bottleneck(inputs, filters, kernel, t, alpha, s, r=False):
    """Bottleneck
    This function defines a basic bottleneck structure.
    # Arguments
        inputs: Tensor, input tensor of conv layer.
        filters: Integer, the dimensionality of the output space.
        kernel: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
        t: Integer, expansion factor.
            t is always applied to the input size.
        s: An integer or tuple/list of 2 integers,specifying the strides
            of the convolution along the width and height.Can be a single
            integer to specify the same value for all spatial dimensions.
        alpha: Integer, width multiplier.
        r: Boolean, Whether to use the residuals.
    # Returns
        Output tensor.
    """

    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    # Depth
    tchannel = K.int_shape(inputs)[channel_axis] * t
    # Width
    cchannel = int(filters * alpha)

    x = _conv_block(inputs, tchannel, (1, 1), (1, 1))

    x = DepthwiseConv2D(kernel, strides=(s, s), depth_multiplier=1, padding='same')(x)
    
    x = BatchNormalization(axis=channel_axis)(x)
    #x = BatchNormalizationF16(axis=channel_axis)(x)
    
    x = LeakyReLU(alpha=0.03)(x)

    x = Conv2D(cchannel, (1, 1), strides=(1, 1), padding='same')(x)
    
    x = BatchNormalization(axis=channel_axis)(x)
    #x = BatchNormalizationF16(axis=channel_axis)(x)
    
    if r:
        x = Add()([x, inputs])

    return x


def _inverted_residual_block(inputs, filters, kernel, t, alpha, strides, n):
    """Inverted Residual Block
    This function defines a sequence of 1 or more identical layers.
    # Arguments
        inputs: Tensor, input tensor of conv layer.
        filters: Integer, the dimensionality of the output space.
        kernel: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
        t: Integer, expansion factor.
            t is always applied to the input size.
        alpha: Integer, width multiplier.
        s: An integer or tuple/list of 2 integers,specifying the strides
            of the convolution along the width and height.Can be a single
            integer to specify the same value for all spatial dimensions.
        n: Integer, layer repeat times.
    # Returns
        Output tensor.
    """

    x = _bottleneck(inputs, filters, kernel, t, alpha, strides)

    for i in range(1, n):
        x = _bottleneck(x, filters, kernel, t, alpha, 1, True)

    return x


def MobileNetv2(input_shape, k, alpha=1.0):
    """MobileNetv2
    This function defines a MobileNetv2 architectures.
    # Arguments
        input_shape: An integer or tuple/list of 3 integers, shape
            of input tensor.
        k: Integer, number of classes.
        alpha: Integer, width multiplier, better in [0.35, 0.50, 0.75, 1.0, 1.3, 1.4].
    # Returns
        MobileNetv2 model.
    """
    inputs = Input(shape=input_shape)

    #first_filters = _make_divisible(32 * alpha, 8)
    first_filters = 16
    x = _conv_block(inputs, first_filters, (3, 3), strides=(2, 2))

    x = _inverted_residual_block(x, 16, (3, 3), t=1, alpha=alpha, strides=1, n=1)
    x = _inverted_residual_block(x, 32, (3, 3), t=1, alpha=alpha, strides=2, n=1)
    x = _inverted_residual_block(x, 64, (3, 3), t=1, alpha=alpha, strides=2, n=1)
    #x = _inverted_residual_block(x, 96, (3, 3), t=3, alpha=alpha, strides=1, n=2)
    x = _inverted_residual_block(x, 96, (3, 3), t=1, alpha=alpha, strides=2, n=1)
    #x = _inverted_residual_block(x, 320, (3, 3), t=6, alpha=alpha, strides=1, n=1)

    #if alpha > 1.0:
        #last_filters = _make_divisible(1280 * alpha, 8)
    #else:
        #last_filters = 1280
    last_filters = 32
    x = _conv_block(x, last_filters, (1, 1), strides=(1, 1))
    x = GlobalAveragePooling2D()(x)
    x = Reshape((1, 1, last_filters))(x)
    x = Dropout(0.3, name='Dropout')(x)
    x = Conv2D(k, (1, 1), padding='same')(x)

    x = Activation('softmax', name='softmax')(x)
    output = Reshape((k,))(x)

    model = Model(inputs, output)
    # plot_model(model, to_file='images/MobileNetv2.png', show_shapes=True)

    return model


if __name__ == '__main__':
    model = MobileNetv2((48, 48, 1), 7, 1.0)
    print(model.summary())


# In[4]:

x = np.load("img_train.npy")
y = np.load("img_label.npy")

x_train = x[0:int(28709*0.8)]/255
y_train = y[0:int(28709*0.8)]
x_val = x[int(28709*0.8): 28709]/255
y_val = y[int(28709*0.8):28709]


# In[5]:

model.compile(loss='categorical_crossentropy',
              optimizer= 'Adam',
              metrics=['accuracy'])
#print(MODEL.model.summary())


# In[6]:

from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=25,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.3,
    zoom_range=[0.8, 1.2],
    horizontal_flip=True,
    vertical_flip=True
    )

datagen.fit(x_train)
train_generator = datagen.flow(x_train,y_train,batch_size = 200)
#learning_rate_function = ReduceLROnPlateau(monitor='val_acc',patience=2,epsilon=0.00001,verbose=1,factor=0.2)
train_history2 = model.fit_generator(train_generator, steps_per_epoch=1500,epochs=25,verbose=1,validation_data=(x_val,y_val))


# In[7]:

model_tmp = model


# In[8]:

import csv
n = 7178
test_x = []
for i in range(n):
    test_x.append([])

count = 0
with open('test.csv', newline='') as csvfile:
    rows = csv.reader(csvfile)
    for row in rows:
        if(row[0] != "id"):
            test_x[count].append(row[1])
            count += 1
x_test = []
for i in range(len(test_x)):
    x_test.append(test_x[i][0].split(' '))

for i in range(len(test_x)):
    for j in range(len(x_test[0])):
        x_test[i][j] = float(x_test[i][j])
x_test = np.array(x_test)
x_test = x_test.reshape(7178,48,48,1)
x_test = x_test/255


# In[9]:

hw3_model = load_model("hw3_model.h5")


# In[10]:

hw3_predict = hw3_model.predict_classes(x_test)


# In[11]:

hw3_y = np_utils.to_categorical(hw3_predict)


# In[17]:

epochs = 3600
batch_size = 3000
model_tmp.fit(x_test, hw3_y,
          batch_size=batch_size,
          epochs=epochs)


# In[18]:

model_tmp.save_weights("weight_train.h5")


# In[19]:

predict_test = model_tmp.predict(x_test)


# In[20]:

prediction = np.argmax(predict_test,axis=1)


# In[21]:

with open("train_sub.csv", 'w', newline='') as csvfile:
    csvfile.write('id,label\n')
    for i, v in enumerate(prediction):
        csvfile.write('%d,%d\n' %(i, prediction[i]))

