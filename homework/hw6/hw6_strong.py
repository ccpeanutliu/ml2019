
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


jieba.load_userdict(sys.argv[2])


# In[3]:


w2v_model = Word2Vec.load("modelwv.model")


# In[4]:


embedding_matrix = np.zeros((len(w2v_model.wv.vocab.items()) + 1, w2v_model.vector_size))
word2idx = {}

vocab_list = [(word, w2v_model.wv[word]) for word, _ in w2v_model.wv.vocab.items()]
for i, vocab in enumerate(vocab_list):
    word, vec = vocab
    embedding_matrix[i + 1] = vec
    word2idx[word] = i + 1


# In[5]:


embedding_layer = Embedding(input_dim=embedding_matrix.shape[0],
                            output_dim=embedding_matrix.shape[1],
                            mask_zero=True,
                            weights=[embedding_matrix],
                            trainable=True)


# In[6]:


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


# In[7]:


count = 0
test_x = []
test_n = 20000
for i in range(test_n):
    test_x.append([])
    
with open(sys.argv[1], newline='',encoding="utf-8") as csvfile:
    rows = csv.reader(csvfile)
    for row in rows:
        if(row[0] != "id"):
            test_x[count].append(row[1])
            count += 1
test_x = np.array(test_x)


# In[8]:


x_test = []
for i in range(test_n):
    x_test.append([])
for i in range(test_n):
    x_test[i].append(jieba.lcut(test_x[i,0]))


# In[9]:


input_x = []
for i in range(test_n):
    input_x.append(x_test[i][0])


# In[10]:


PADDING_LENGTH = 100
input_X = text_to_index(input_x)
input_X = pad_sequences(input_X, maxlen=PADDING_LENGTH)


# In[11]:

#wget.download("https://github.com/cphuamao/ML2019SPRING/releases/download/hw6_strong_model/hw6_strong_model.h5", out="hw6_strong_model.h5")
model = load_model("hw6_strong_model.h5")


# In[12]:


Y_preds = model.predict(input_X)
print("Shape:", Y_preds.shape)
print("Sample:", Y_preds[0])


# In[16]:


Y_preds[Y_preds > 0.5] = 1
Y_preds[Y_preds <= 0.5] = 0
Y_preds_label = np.uint8(Y_preds)
print("Shape:", Y_preds_label.shape)
print("Sample:", Y_preds_label[0])


# In[17]:


with open(sys.argv[3], 'w', newline='') as csvfile:
    csvfile.write('id,label\n')
    for i, v in enumerate(Y_preds_label):
        csvfile.write('%d,%d\n' %(i, Y_preds_label[i]))

