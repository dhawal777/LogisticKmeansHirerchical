#!/usr/bin/env python
# coding: utf-8

# In[14]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import classification_report, confusion_matrix ,accuracy_score
import copy


# In[15]:


df=pd.read_csv("wine-quality/data.csv")


# In[16]:


X =df.drop(['quality'],axis=1)
y=df['quality']
# df
X = (X - X.mean())/X.std()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
intrain=X_train
intest=X_test


# In[17]:


def grad(thetaT):
    return 1/(1+np.exp(-thetaT))
    


# In[18]:


df1=pd.concat([X_train,y_train],axis=1)


# In[19]:


def gradientDescent(X,y,theta,iters,alpha):
    for i in range(iters):
        theta = theta - (alpha/len(X)) * np.sum(X * (grad(X @ theta.T) - y), axis=0)
    
    return theta


# **ONE VS ONE**
# 
# * Extracting data corresponding to two feature at a time and generating theta values for it
# * Then test on whole betaList

# In[20]:


betaList=[]
count=[]
for j in range(4,10):
    for k in range(j+1,10):
        df2=copy.deepcopy(df1)
        df3=df2.loc[df2['quality'].isin([j,k])]
        X=df3
        X=df3.drop(['quality'],axis=1)
        ones = np.ones([X.shape[0],1])
        X=X.values
        X = np.concatenate((ones,X),axis=1)
        y=pd.DataFrame(df3['quality'])
        y=y.values
        y_temp=[]
        for i in range(len(y)):
            if y[i]==j:
                y_temp.append(1)
            else:
                y_temp.append(0)

        y_temp1=pd.DataFrame(y_temp)
        y_temp2=y_temp1.values
        theta = np.zeros([1,12])
        alpha = 0.01
        iters = 1000
        g = gradientDescent(X,y_temp2,theta,iters,alpha)
        betaList.append(g[0])


# *Follow the same order of class for testing betaList corresponds to that and if >0.5 than j else k and out of all and that class that will be most dominating will be final class for that test data*

# In[21]:


y_pred=[]
for index,row in X_test.iterrows():
    dict1={}
    for i in range(4,10):
        dict1[i]=0
    max1=0
    row=list(row)
    class1=0
    r=0
    for i in range(4,10):
        for k in range(i+1,10):
            y1=0
            for j in range(1,12):
                y1=y1+betaList[r][j]*row[j-1]
            
            y1=y1+betaList[r][0]
            y1=grad(y1)
            if(y1>=0.5):
                dict1[i]=dict1[i]+1
            else:
                dict1[k]=dict1[k]+1
            r=r+1
    for i in range(4,10):
        if(dict1[i]>=max1):
            max1=dict1[i]
            class1=i
    y_pred.append(class1)


# In[22]:


print(confusion_matrix(y_test, y_pred))  
print("Accuracy: ",accuracy_score(y_test,y_pred))
print((y_test==y_pred).mean())


# In[23]:


from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
model = LogisticRegression(solver = 'lbfgs',multi_class='multinomial',max_iter=10000)
intrain1 = StandardScaler().fit_transform(intrain)


# In[24]:


model.fit(intrain1, y_train)
intest1=StandardScaler().fit_transform(intest)
y_pred = model.predict(intest1)


# In[25]:


count_misclassified = (y_test != y_pred).sum()
accuracy = accuracy_score(y_test, y_pred)
print('System Accuracy: {:.2f}'.format(accuracy))

