#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import classification_report, confusion_matrix ,accuracy_score


# In[2]:


df=pd.read_csv("AdmissionDataset/data.csv")
X =df.drop(['Chance of Admit ','Serial No.'],axis=1)
y=df['Chance of Admit ']


# In[3]:


X = (X - X.mean())/X.std()
y=list(y)
for i in range(len(y)):
    if y[i]>0.5:
        y[i]=1
    else:
        y[i]=0
y=pd.DataFrame({'Chance of Admit ':y
    
})


# In[4]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
features=['GRE Score','TOEFL Score','University Rating','SOP','LOR ','CGPA','Research']
intrain=X_train
intest=X_test


# **Sigmoid function**

# In[5]:


def grad(thetaT):
    return 1/(1+np.exp(-thetaT))
    


# In[6]:


my_data=pd.concat([X_train,y_train],axis=1)
X=X_train
ones = np.ones([X.shape[0],1])
X = np.concatenate((ones,X),axis=1)
y=pd.DataFrame(y_train)
y=y.values
theta = np.zeros([1,8])
alpha = 0.01
iters = 1000


# **Applying gradient descent to get min cost based theta values**

# In[7]:


def gradientDescent(X,y,theta,iters,alpha):
    for i in range(iters):
        theta = theta - (alpha/len(X)) * np.sum(X * (grad(X @ theta.T) - y), axis=0)
    
    return theta


# In[8]:


g = gradientDescent(X,y,theta,iters,alpha)
betaList=g[0]
ones = np.ones([X_test.shape[0],1])
X_test = np.concatenate((ones,X_test),axis=1)


# In[9]:


y_pred=grad(X_test@betaList)


# In[10]:


y_pred=list(y_pred)
df=pd.DataFrame({'Admit':y_pred})


# In[11]:


df[df['Admit']<0.5]=0
df[df['Admit']>=0.5]=1
df
y_pred=list(df['Admit'])


# In[12]:


print(confusion_matrix(y_test, y_pred))  
print("System Accuracy: ",accuracy_score(y_test,y_pred))
print((y_test==y_pred).mean())


# In[13]:


from sklearn.linear_model import LogisticRegression
logisticRegr = LogisticRegression(solver = 'lbfgs')
logisticRegr.fit(intrain, y_train['Chance of Admit '])
y_pred = logisticRegr.predict(intest)
# print(len(y_pred))
# print(len(y_test))
print(confusion_matrix(y_test, y_pred))  
# print(classification_report(y_test, y_pred)) 
score = logisticRegr.score(intest, y_test)
print(score)
# print((y_test==y_pred).mean())

