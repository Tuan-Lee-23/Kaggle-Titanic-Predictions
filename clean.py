#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_style("whitegrid")


# In[56]:


train = pd.read_csv("train.csv")


# In[57]:


train.head()


# In[58]:


plt.figure(figsize = (10,7))
sns.heatmap(train.isnull(), cbar = False, cmap = "viridis")


# In[59]:


sns.distplot(train['Age'].dropna())


# In[60]:


sns.boxplot(x = 'Pclass', y = 'Age', data = train)


# In[ ]:





# In[61]:


class_mean = np.zeros(3)

for i in range(0,3):
    class_mean[i] = train[train['Pclass'] == i + 1]['Age'].mean()
    print("Class ", i + 1, ": ", class_mean.item(i))


# In[62]:


def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):
        if Pclass == 1:
            return class_mean.item(0)
        elif Pclass == 2:
            return class_mean.item(1)
        elif Pclass == 3:
            return class_mean.item(2)
    else:
        return Age


# In[63]:


train['Age'] = train[['Age','Pclass']].apply(impute_age, axis = 1)


# In[69]:


train.drop("Cabin", inplace = True, axis = 1)


# In[103]:


train.head()


# In[106]:


sex = pd.get_dummies(train['Sex'], drop_first=True)
pclass = pd.get_dummies(train['Pclass'], drop_first=True)
embarked = pd.get_dummies(train['Embarked'], drop_first=True)

train_labeled = pd.concat([train, sex, pclass, embarked], axis = 1)
train_labeled.head()


# In[ ]:





# In[127]:


train_labeled.drop("Embarked", axis = 1, inplace = True)


# In[130]:


x = train_labeled.drop("Survived", axis = 1)
y = train_labeled["Survived"]
x


# In[131]:


from sklearn.linear_model import LogisticRegression


# In[132]:


logmodel = LogisticRegression()


# In[133]:


logmodel.fit(x, y)


# In[ ]:




