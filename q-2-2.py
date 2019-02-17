#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import classification_report, confusion_matrix ,accuracy_score
import matplotlib
import matplotlib.pyplot as plt
import copy


# In[9]:


df=pd.read_csv("AdmissionDataset/data.csv")
    # df[df['Chance of Admit ']<0.5]=0
    # df[df['Chance of Admit ']>=0.5]=1
X =df.drop(['Chance of Admit ','Serial No.'],axis=1)
y=df['Chance of Admit ']
# df
thres=0.5

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
knntrain=copy.deepcopy(X_train)
knntest=copy.deepcopy(X_test)
knnytrain=copy.deepcopy(y_train)
knnytest=copy.deepcopy(y_test)
X_train = (X_train - X_train.mean())/X_train.std()
X_test = (X_test - X_test.mean())/X_test.std()
y_train=list(y_train)
for i in range(len(y_train)):
    if y_train[i]>thres:
        y_train[i]=1
    else:
        y_train[i]=0
y_train=pd.DataFrame({'Chance of Admit ':y_train

})
y_test=list(y_test)
for i in range(len(y_test)):
    if y_test[i]>thres:
        y_test[i]=1
    else:
        y_test[i]=0
y_test=pd.DataFrame({'Chance of Admit ':y_test

})

features=['GRE Score','TOEFL Score','University Rating','SOP','LOR ','CGPA','Research']
#     intrain=X_train
#     intest=X_test


# In[10]:


def logistic(X_train,X_test,y_train,y_test):
#     thres=0.4
    
    # y_train

    def grad(thetaT):
        return 1/(1+np.exp(-thetaT))


    my_data=pd.concat([X_train,y_train],axis=1)

    X=X_train

    ones = np.ones([X.shape[0],1])


    X = np.concatenate((ones,X),axis=1)



    y=pd.DataFrame(y_train)
    y=y.values
    # y

    theta = np.zeros([1,8])


    alpha = 0.01

    iters = 1000

    def gradientDescent(X,y,theta,iters,alpha):
        for i in range(iters):
            theta = theta - (alpha/len(X)) * np.sum(X * (grad(X @ theta.T) - y), axis=0)

        return theta

    #running the gd and cost function

    g = gradientDescent(X,y,theta,iters,alpha)
    betaList=g[0]
    ones = np.ones([X_test.shape[0],1])
    X_test = np.concatenate((ones,X_test),axis=1)

    y_pred=grad(X_test@betaList)

    y_pred=list(y_pred)
#     print(y_pred)
    df=pd.DataFrame({'Admit':y_pred})

    df[df['Admit']<=thres]=0
    df[df['Admit']>thres]=1
#     print(df)
    y_pred=list(df['Admit'])


#     print(confusion_matrix(y_test, y_pred))  
    print("-------------------Logistic Accuracy Score-------------------------")
    print("Accuracy: ",accuracy_score(y_test,y_pred))


# In[11]:


def euclidean_distance(x, y): 
#row by row i.e all features resultant euclid distance in single line
    return np.sqrt(np.sum((x - y) ** 2))


# In[12]:


def euclid(k,X_train,X_test,y_train,y_test):
    y_res=[]
    for index,row in X_test.iterrows():
        result=[]
        for index1,row1 in X_train.iterrows():
                result.append(euclidean_distance(row,row1))
        df1=pd.DataFrame(
        {
            'dist':result,
            'class':y_train
        })
#         print(df1)
        df1=df1.sort_values(by=['dist'])
        count=0;
#         k=5
        classVote={}
        max1=0
        res=""
        for index1,row1 in df1.iterrows():
            capital=row1['class']
            if capital > thres:
                capital=1
            else:
                capital=0
            count=count+1
            if  capital in classVote:
                classVote[capital] = classVote[capital]+1
            else:
                classVote[capital]=1
            if classVote[capital]>=max1:
                res=capital
                max1=classVote[capital]
            if count==k:
                break;
        y_res.append(res)
    return y_res,y_test


# In[13]:


logistic(X_train,X_test,y_train,y_test)


# In[14]:


y_res,y_test=euclid(5,knntrain,knntest,knnytrain,knnytest)
y_test=list(y_test)
for i in range(len(y_test)):
    if y_test[i]>thres:
        y_test[i]=1
    else:
        y_test[i]=0
print("---------------------KNN AccuracyScore------------------------")
print("Accuracy: ",accuracy_score(y_test,y_res))

