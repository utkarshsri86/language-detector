#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
import pandas as pd 


# In[23]:


data =pd.read_csv("language (1).csv")


# In[24]:


data


# In[19]:


from sklearn.feature_extraction.text import CountVectorizer


# In[20]:


from sklearn.model_selection import train_test_split


# In[21]:


from sklearn.naive_bayes import MultinomialNB


# In[26]:


data.isnull().sum()


# In[27]:


data['language'].value_counts()


# In[32]:


x=np.array(data['Text'])
y=np.array(data['language'])


# In[38]:


cv =CountVectorizer()
x = cv.fit_transform(x)


# In[39]:


x_train,x_test,y_train,y_test= train_test_split(x,y,test_size =0.33 ,random_state =42)


# In[40]:


x_train


# In[41]:


print(x_train)


# In[42]:


y_test


# In[46]:


model = MultinomialNB()


# In[47]:


model.fit(x_train,y_train)


# In[49]:


model.score(x_test,y_test)


# In[63]:


user = input("enter a text")
data =cv.transform([user]).toarray()
output = model.predict(data)
print(output)


# In[ ]:




