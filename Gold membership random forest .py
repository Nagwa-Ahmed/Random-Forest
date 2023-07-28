#!/usr/bin/env python
# coding: utf-8

# In[18]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas  as pd 
import numpy as np
import os
from matplotlib import pyplot


# In[2]:


os.chdir('C:\\Users\\Compu Tech\\Desktop\\El araby internship')


# In[3]:


data=pd.read_excel('market data random forest.xlsx')


# In[4]:


data.head()


# In[5]:


data.columns


# In[8]:


X=data.values[:,0:23]
Y=data.values[:,-1]


# In[9]:


X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=100)


# In[10]:


X_train


# In[11]:


clf=RandomForestClassifier(n_jobs=2,random_state=100)


# In[12]:


clf.fit(X_train,y_train)


# In[13]:


y_pred=clf.predict(X_test)


# In[14]:


print ("Accuracy is "), accuracy_score(y_test,y_pred)*100


# In[16]:


importances_sk = clf.feature_importances_


# In[19]:


model=clf
importance = model.feature_importances_
colnames=list(data.columns)
# summarize feature importance
for i,v in enumerate(importance):
    print(colnames[i])
    print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




