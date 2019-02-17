#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import sklearn as sk
from sklearn.model_selection import train_test_split
eps=np.finfo(float).eps
from binarytree import tree,Node
from sklearn.metrics import classification_report, confusion_matrix ,accuracy_score
from sklearn.tree import DecisionTreeClassifier
from operator import itemgetter
import copy
import collections
from pylab import *
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA as PCA
import sys
import copy
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
import operator


# In[2]:


def mean(numbers):
    return sum(numbers)/float(len(numbers))


# In[3]:


def stdev(numbers):
    avg = mean(numbers)
    varianceNum=0.0
    for x in numbers:
        varianceNum=varianceNum+pow(x-avg,2)
    variance=varianceNum/float(len(numbers)-1)
    return math.sqrt(variance)


# In[4]:


df=pd.read_csv("Train_data.csv")
df
df.drop(['num_outbound_cmds'],inplace=True,axis=1)
df1=copy.deepcopy(df)
col=list(df)
mean1={}
len1=len(col)
len1
# df['num_outbound_cmds']


# In[5]:


labelencoder = LabelEncoder()
df['protocol_type'] = labelencoder.fit_transform(df['protocol_type'])
df1['protocol_type'] = labelencoder.fit_transform(df1['protocol_type'])
df


# In[6]:


for i in col:
    if i!="xAttack":
        mean1[i]=mean(df[i])
std1={}
for i in col:
    if i!="xAttack":
        std1[i]=stdev(df[i])
for i in col:
    if i!='xAttack':
        df[i]=(df[i]-mean1[i])/std1[i]


# In[7]:


X = df.iloc[:,0:40].values
y = df.iloc[:,40].values
X_std=X
mean_vec = np.mean(X_std, axis=0)
# print(mean_vec)


# In[8]:


cov_mat = (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0]-1)
eig_vals, eig_vecs = np.linalg.eig(cov_mat)
eig_pairs=[]
dict1={}
for i in range(len(eig_vals)):
    t=(np.abs(eig_vals[i]), eig_vecs[:,i])
    eig_pairs.append(t)
    dict1[col[i]]=eig_vals[i]
total=0
eig_pairs.sort()
eig_pairs.reverse()
error=0
for i in eig_pairs:
    total+=i[0]
j=0;
arr=[]
for i in eig_pairs:
    error+=((i[0]/total)*100)
    arr.append(i[1])
#     print 
    j=j+1;
    if(error>90):
        break;
arr=np.array(arr)
X1=X_std.dot(arr.T)
X1
l=[];
x=len(X1);
for i in range(0,x):
    l.append(i)
Df = pd.DataFrame(data = X1,
                  index=l)


# In[9]:


sorted_dict=sorted(dict1.items(), key=lambda kv: kv[1],reverse=True)
sorted_dict


# In[10]:


Df


# In[11]:


X1 = df1.iloc[:,0:40].values
y1 = df1.iloc[:,40].values
x = StandardScaler().fit_transform(X1)
pca = PCA(n_components=j)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents)


# In[12]:


principalDf


# **Reasons**
# 
# * Data Contain equal amount of numerical and categorical feature and only normal PCA is not sufficient to handle this and as we can observe in printed dictionary according to sorted eigen values most of the categorical data will not be able to dominate and even if selected the value of categorical data has no meaning
# * PCA always considered the low variance components in the data as noise and recommend us to throw away those components but for categorical we can't analyse variance and are thus unnecessarily eliminated.
# * PCA has problems when there is heavy bias in the sampled data and we can observed many attribute are heavily baised toward a particular value.
# 
# 
