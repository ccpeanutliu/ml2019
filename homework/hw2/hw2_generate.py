
# coding: utf-8

# In[1]:


import numpy as np
import sys
import csv
from numpy.linalg import inv


def standalize(x):
    minn = np.min(x, axis=0)
    maxx = np.max(x, axis=0)
    meann = np.mean(x, axis=0)
    stdd = np.std(x, axis=0)
    for i in range(stdd.shape[1]):
        if stdd[0,i] == 0:
            stdd[0,i] = 1
    for i in range(x.shape[0]):
#         print(x[i] - minn, maxx-minn)
        x[i] = (x[i] - meann)/stdd
    return x



ft = open(sys.argv[5],"r")

w = np.load("hw2_weight_gen.npy")
# In[21]:
b = np.load("hw2_bias.npy")

linet = ft.readline()
linet = ft.readline()


# In[22]:


test = []
for i in range(16281):
    test.append([])


# In[23]:


for i in range(16281):
    linet = linet.split(',')
    for j in range(len(linet)):
        test[i].append(float(linet[j]))
    linet = ft.readline()


# In[24]:


test = np.matrix(test)
standalize(test)


ans = []
for i in range(16281):
    ans.append([])


# In[27]:


for i in range(test.shape[0]):
    z = np.dot(test[i,:],w) + b
    ans[i].append((1.0/(1 + np.exp(-z)))[0,0])


# In[28]:


len(ans)


# In[29]:


for i in range(len(ans)):
    if ans[i][0] > 0.5:
        ans[i][0] = 0
    else:
        ans[i][0] = 1


# In[30]:


with open(sys.argv[6], 'w', newline='') as csvfile:
  csvfile.write('id,label\n')
  for i, v in enumerate(ans):
    csvfile.write('%d,%d\n' %(i+1, ans[i][0]))


# In[32]:
