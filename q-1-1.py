#!/usr/bin/env python
# coding: utf-8

# * In order to handle “curse of dimensionality” and avoid issues like over-fitting in high dimensional space, methods like Principal Component analysis is used. 
# * This method combines highly correlated variables together to form a smaller number of an artificial set of variables which is called “principal components” that account for most variance in the data.

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


# **Calculating mean and variance**

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


# **It is important to normalize the predictor as original predictors can be on the different scale and can contribute significantly towards variance**

# In[4]:


df=pd.read_csv("intrusion_detection/data.csv")
df
df1=copy.deepcopy(df)
col=list(df)
mean1={}
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


# calculate the covariance matrix for the data which would measure how two predictors move together.
# calculate Eigen values and Eigen vector of the above matrix. This helps in finding underlying patterns in the data.(eigen vector for direction and eigen value show domination of a atribute)

# In[5]:


X = df.iloc[:,0:29].values
y = df.iloc[:,29].values
X_std=X
meanVector = np.mean(X_std, axis=0)
covMat = (X_std - meanVector).T.dot((X_std - meanVector)) / (X_std.shape[0]-1)
eigValues, eigVectors = np.linalg.eig(covMat)
eig_pairs=[]
for i in range(len(eigValues)):
    t=(np.abs(eigValues[i]), eigVectors[:,i])
    eig_pairs.append(t)
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


# In[6]:


Df


# In[7]:


X1 = df1.iloc[:,0:29].values
y1 = df1.iloc[:,29].values
x = StandardScaler().fit_transform(X1)
pca = PCA(n_components=j)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents)


# In[8]:


principalDf

