#!/usr/bin/env python
# coding: utf-8

# # TASK 1 LETSGROWMORE
# 

# In[5]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection  import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score


# In[6]:


df=pd.read_csv("IRIS.csv")
df.head()


# In[7]:


df.tail()


# In[8]:


df.shape


# In[9]:


df.isnull()


# In[10]:


df.isnull().sum() #checking sum of all missing values present in the dataset


# In[11]:


df.nunique()


# In[12]:


df.Species.unique()


# In[13]:


df.Species.value_counts()


# In[14]:


df.max()


# In[15]:


df.min()


# In[16]:


df.drop('Id',axis=1,inplace=True)
df.head()


# # visualization

# In[17]:


sns.boxplot(x='Species',y='PetalLengthCm',data=df)
plt.show()


# In[18]:


sns.boxplot(x='Species',y='SepalWidthCm',data=df)
plt.show()


# In[19]:


sns.boxplot(x='Species',y='SepalLengthCm',data=df)
plt.show()


# In[20]:


sns.boxplot(x='Species',y='PetalWidthCm',data=df)
plt.show()


# In[21]:


sns.boxplot(y='SepalLengthCm',data=df)
plt.show()


# In[22]:


sns.boxplot(y='SepalWidthCm',data=df)
plt.show()


# In[23]:


sns.boxplot(y='PetalLengthCm',data=df)
plt.show()


# In[24]:


sns.boxplot(y='PetalWidthCm',data=df)
plt.show()


# In[25]:


#a pairplot plot a pairwise  relationship in a dataset

sns.pairplot(df,hue='Species')


# # data preprocessing  corelation matrix

# In[26]:



sns.heatmap(df.corr(),annot=True,cmap="seismic")
plt.show()


# # label encoder

# In[27]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()


# In[28]:


df['Species']=le.fit_transform(df['Species'])
df.head()


# In[29]:


x=df.drop(columns=['Species'])
y=df['Species']
x[:5]


# In[30]:


y[:5]


# # splitting the dataset into training set and test set

# In[31]:


from sklearn.model_selection  import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)


# In[32]:


from sklearn.linear_model import LinearRegression
model=LinearRegression()


# In[33]:


model.fit(x,y)


# In[34]:


model.score(x,y)


# In[35]:


model.coef_


# In[36]:


model.intercept_


# # making prdeictions

# In[37]:


y_pred=model.predict(x_test)


# In[38]:


print("mean squared error: %.2f" %np.mean((y_pred-y_test)**2))


# In[ ]:




