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
data=Df.values
k=5
c = data.shape[1]


# In[5]:


mean = np.mean(data,axis = 0)
std = np.std(data, axis = 0)


# In[6]:


centers = np.random.randn(k,c)*std+mean


# In[7]:


n=data.shape[0]


# In[8]:


def euclidean_distance(x, y): 
        return np.sqrt(np.sum((x - y) ** 2))


# In[9]:


centers_old = np.zeros(centers.shape)
centers_new = copy.deepcopy(centers)
clusters = np.zeros(n)
distances = np.zeros((n,k))
error=0
for i in range(k):
    error = error+euclidean_distance(centers_new[i],centers_old[i])


# * Assigning all points to current cluster center.
# * And than transforming the center according to assignment of points
# * Repeat step 1 and 2 till error==0 or itr<1000
# 

# In[10]:


itr=0
while error!=0:
    for i in range(k):
        #getting distance of all point wrt to each cluster center
        distances[:,i] = np.linalg.norm(data - centers_new[i], axis=1)
    #assigning the cluster to each point whose distance is minimum
    clusters = np.argmin(distances, axis = 1)
    centers_old = copy.deepcopy(centers_new)
    for i in range(k):
        #if cluster has no point than reinitialize the process
        if(len(data[clusters==i])==0):
            centers = np.random.randn(5,14)
            for j in range(k):
                centers[j]=centers[j]*std+mean
            centers_old = np.zeros(centers.shape)
            centers_new = copy.deepcopy(centers)
            clusters = np.zeros(n)
            distances = np.zeros((n,k))
            temp=0
            for i in range(k):
                temp = temp+euclidean_distance(centers_new[i],centers_old[i])
            error=temp
            itr=0
            break;
        else:
            #assigning new center base on assignment of points i.e mean of all the points that belog to cluster==i
            centers_new[i] = np.mean(data[clusters == i], axis=0)
    temp=0
    for i in range(k):
        temp = temp+euclidean_distance(centers_new[i],centers_old[i])
    error=temp
    itr=itr+1


# In[11]:


labelencoder = LabelEncoder()
y = labelencoder.fit_transform(y)

ans=[]
ans1=[]
for i in range(k):
    ans.append(np.count_nonzero(clusters==i))
    ans1.append(np.count_nonzero(y==i))
# print(ans)
# print(ans1)


# In[12]:


def purityScore(y_true, y_pred):
    contingency_matrix =metrics.cluster.contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix) 


# In[13]:


print("--------------Purity of Kmeans----------------------")
print(purityScore(y,clusters))


# In[14]:


X1 = df1.iloc[:,0:29].values
y1 = df1.iloc[:,29].values
x = StandardScaler().fit_transform(X1)
pca = PCA(n_components=14)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents)
X2=principalDf.values
kmeans = KMeans(n_clusters=5)
kmeans = kmeans.fit(X2)
labels = kmeans.predict(X2)
ans=[]
ans1=[]
for i in range(k):
    ans.append(np.count_nonzero(labels==i))
    ans1.append(np.count_nonzero(y==i))


# In[15]:


print("--------------Purity of Kmeans System----------------------")
print(purityScore(y,labels))

