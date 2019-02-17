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
from sklearn import mixture


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


# In[4]:


X = df.iloc[:,0:29].values
y = df.iloc[:,29].values
X_std=X
mean_vec = np.mean(X_std, axis=0)
cov_mat = (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0]-1)
eig_vals, eig_vecs = np.linalg.eig(cov_mat)
eig_pairs=[]
for i in range(len(eig_vals)):
    t=(np.abs(eig_vals[i]), eig_vecs[:,i])
    eig_pairs.append(t)
total=0
eig_pairs.sort()
eig_pairs.reverse()
for i in eig_pairs:
    total+=i[0]    
error=0
j=0;
arr=[]
for i in eig_pairs:
    error+=((i[0]/total)*100)
    arr.append(i[1])
    print 
    j=j+1;
    if(error>90):
        break;
arr=np.array(arr)
X1=X_std.dot(arr.T)
X1
l=[];
x=len(X1);
# print(x)
for i in range(0,x):
    l.append(i)
Df = pd.DataFrame(data = X1,
                  index=l)
data=Df.values
k=5
c = data.shape[1]


# In[5]:


def purity_score(y_true, y_pred):
    contingency_matrix =metrics.cluster.contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


# **GMM VS KMEANS**
# * GMM is a lot more flexible in terms of cluster covariance.
# * k-means is actually a special case of GMM in which each cluster’s covariance along all dimensions approaches 0.   This implies that a point will get assigned only to the cluster closest to it.
# * With GMM, each cluster can have unconstrained covariance structure. Think of rotated and/or elongated   distribution of points in a cluster, instead of spherical as in kmeans. As a result, cluster assignment is much   more flexible in GMM than in k-means
# *In kmeans, a point belongs to one and only one cluster, whereas in GMM a point belongs to each cluster to a different degree. The degree is based on the probability of the point being generated from each cluster’s (multivariate) normal distribution, with cluster center as the distribution’s mean and cluster covariance as its covariance. 

# In[6]:


gm=mixture.GaussianMixture(n_components=5).fit(data)
labels = gm.predict(data)
print("----------Gaussian Purity Score------------------")
print(purity_score(y,labels))

