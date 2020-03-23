
# coding: utf-8

# In[6]:


import numpy as np
import sys
import csv

# In[12]:


def Loss(y, fx):
    dd =np.multiply(y,np.log(fx)) + np.multiply((1-y),np.log(1- fx))
    return -np.mean(dd)
def sigmoid(x, derivative=False):
    return x*(1-x) if derivative else 1/(1+np.exp(-x))
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


# In[1]:


ft = open(sys.argv[5],"r")


# In[2]:


linet = ft.readline()
linet = ft.readline()


# In[3]:


test = []
for i in range(16281):
    test.append([])


# In[4]:


for i in range(16281):
    linet = linet.split(',')
    for j in range(len(linet)):
        test[i].append(float(linet[j]))
    linet = ft.readline()


# In[9]:


weight = np.load('hw2_weight_log.npy')


# In[10]:


test = np.matrix(test)
wT = weight.T
w = weight
mtest = np.mean(test,axis=0)
stest = np.mean(test,axis=0)


# In[13]:


standalize(test)
test_base = test

test = np.hstack((test, np.multiply(test,test)))
test = np.hstack((test, np.ones(shape=(test.shape[0],1))))


# In[14]:


zt = np.dot(test,weight)


# In[15]:


ans = []
for i in range(16281):
    ans.append([])


# In[17]:


ans = sigmoid(zt)


# In[18]:


out = []
for i in range(ans.shape[0]):
    out.append([])
for i in range(ans.shape[0]):
    if ans[i,0] > 0.5:
        out[i].append(1)
    else:
        out[i].append(0)


# In[19]:


wout = []
for i in range(weight.shape[0]):
    wout.append([])
for i in range(weight.shape[0]):
    wout[i].append(weight[i,0])


# In[21]:


import csv
wi = 1
with open(sys.argv[6], 'w', newline='') as csvfile:
  csvfile.write('id,label\n')
#  writer = csv.writer(csvfile)
#  writer.writerows("id","label")
  for i, v in enumerate(out):
    csvfile.write('%d,%d\n' %(i+1, out[i][0]))
