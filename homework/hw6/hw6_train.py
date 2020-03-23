
# coding: utf-8

# In[1]:


import csv
import keras
import numpy as np
import jieba
import sys
from gensim.models.word2vec import Word2Vec
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Embedding, GRU, Dense, LSTM, Dropout, SimpleRNN, Bidirectional
from keras.layers.normalization import BatchNormalization


# In[2]:


jieba.load_userdict(sys.argv[4])


# In[3]:


n = 120000
train_x = []
for i in range(n):
    train_x.append([])
train_y = []


# In[4]:


count = 0
with open(sys.argv[1], newline='',encoding="utf-8") as csvfile:
    rows = csv.reader(csvfile)
    for row in rows:
        if(row[0] != "id"):
            train_x[count].append(row[1])
            count += 1

with open(sys.argv[2], newline='') as csvfile:
    rows = csv.reader(csvfile)
    for row in rows:
        if(row[0] != "id"):
            train_y.append(row[1])


# In[5]:


train_x = np.array(train_x)
train_y = np.array(train_y)


# In[6]:


train_y = train_y.astype("uint8")


# In[7]:


train_y = train_y[0:119017]


# In[8]:


n = 119017
use_x = []
for i in range(n):
    use_x.append([])
for i in range(n):
    use_x[i].append(jieba.lcut(train_x[i,0]))


# In[9]:


count = 0
test_x = []
test_n = 20000
for i in range(test_n):
    test_x.append([])
    
with open(sys.argv[3], newline='',encoding="utf-8") as csvfile:
    rows = csv.reader(csvfile)
    for row in rows:
        if(row[0] != "id"):
            test_x[count].append(row[1])
            count += 1
test_x = np.array(test_x)


# In[10]:


x_test = []
for i in range(test_n):
    x_test.append([])
for i in range(test_n):
    x_test[i].append(jieba.lcut(test_x[i,0]))


# In[11]:


x = []
xx = []
for i in range(n):
    x.append(use_x[i][0])
    xx.append(use_x[i][0])
for i in range(20000):
    xx.append(x_test[i][0])


# In[12]:


np.random.seed(1337)


# In[ ]:


modelwv = Word2Vec(xx,min_count = 4, size=200, sg = 1)#, window = 20, negative = 3)


# In[ ]:


w2v_model = modelwv


# In[ ]:


embedding_matrix = np.zeros((len(w2v_model.wv.vocab.items()) + 1, w2v_model.vector_size))
word2idx = {}

vocab_list = [(word, w2v_model.wv[word]) for word, _ in w2v_model.wv.vocab.items()]
for i, vocab in enumerate(vocab_list):
    word, vec = vocab
    embedding_matrix[i + 1] = vec
    word2idx[word] = i + 1


# In[ ]:


embedding_layer = Embedding(input_dim=embedding_matrix.shape[0],
                            output_dim=embedding_matrix.shape[1],
                            mask_zero=True,
                            weights=[embedding_matrix],
                            trainable=True)


# In[ ]:


def text_to_index(corpus):
    new_corpus = []
    for doc in corpus:
        new_doc = []
        for word in doc:
            try:
                new_doc.append(word2idx[word])
            except:
                new_doc.append(0)
        new_corpus.append(new_doc)
    return np.array(new_corpus)


# In[ ]:


PADDING_LENGTH = 100
X = text_to_index(x)
X = pad_sequences(X, maxlen=PADDING_LENGTH)
print("Shape:", X.shape)
print("Sample:", X[100])


# In[ ]:


def new_model():
    model = Sequential()
    model.add(embedding_layer)
    model.add(GRU(64,activation='sigmoid',inner_activation='hard_sigmoid',return_sequences=True))
    model.add(Dropout(0.5))
    model.add(GRU(128))
    model.add(Dropout(0.5))
    model.add(Dense(16, activation='sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


# In[ ]:


model = new_model()
model.summary()


# In[ ]:


ms = keras.callbacks.ModelCheckpoint("ModelCheck2.h5", monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=False, mode='max', period=1)
model.fit(x=X, y=train_y, batch_size=3000, epochs=10, validation_split=0.2,callbacks=[ms])#,shuffle = True)


# In[23]:


input_x = []
for i in range(test_n):
    input_x.append(x_test[i][0])


# In[24]:


PADDING_LENGTH = 100
input_X = text_to_index(input_x)
input_X = pad_sequences(input_X, maxlen=PADDING_LENGTH)


# In[25]:


Y_preds = model.predict(input_X)
print("Shape:", Y_preds.shape)
print("Sample:", Y_preds[0])


# In[26]:


Y_preds[Y_preds > 0.5] = 1
Y_preds[Y_preds <= 0.5] = 0
Y_preds_label = np.uint8(Y_preds)
print("Shape:", Y_preds_label.shape)
print("Sample:", Y_preds_label[0])


# In[28]:


with open("hw6_output_train.csv", 'w', newline='') as csvfile:
    csvfile.write('id,label\n')
    for i, v in enumerate(Y_preds_label):
        csvfile.write('%d,%d\n' %(i, Y_preds_label[i]))

