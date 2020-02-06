#!/usr/bin/env python
# coding: utf-8

# In[65]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# In[66]:


ds = pd.read_csv('http://www.stat.ufl.edu/~winner/data/airq402.dat',delim_whitespace=1,header = None)


# In[29]:


ds.head()


# In[30]:


ds.shape


# In[31]:


ds.columns = ['City1','City2','Average Fare','Distance','Average weekly passengers', 'market leading airline', 'market share', 'average fare', 'low price airline', 'market share ', 'price']


# In[32]:


ds.head()


# In[33]:


sns.pairplot(ds)


# In[34]:


#Remove outliers from the dat


# In[35]:


def outlier(col):
        q3 = ds[col].quantile(0.75)
        q1 = ds[col].quantile(0.25)
        iqr = q3 - q1
        lowval = q1 - 1.5* iqr
        highval = q3 + 1.5 * iqr
        loc_ret = ds.loc[(ds[col] > lowval) & (ds[col] < highval)]
        return loc_ret


# In[36]:


ds.dtypes


# In[37]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


# In[38]:


ds['market leading airline'] = le.fit_transform(ds['market leading airline'])
ds['City1'] = le.fit_transform(ds['City1'])
ds['City2'] = le.fit_transform(ds['City2'])
ds['low price airline'] = le.fit_transform(ds['low price airline'])


# In[39]:


ds.describe(include = 'all').transpose()


# In[ ]:





# In[40]:


ds.head()


# In[41]:


for col in ds.columns:
    ds[col] = ds[col].astype(int)


# In[42]:


ds.shape


# In[43]:


for col in ds.columns:
    ds = outlier(col)
    


# In[44]:


ds.shape


# In[45]:


sns.pairplot(ds)


# In[46]:


#Drop the independent variables which has less than 0.1 correlation with the dependent variable


# In[47]:


ds.corr()


# In[48]:


## Correlation Output
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
sns.heatmap(ds.corr(), annot=True, vmin=-1, vmax=1, cmap='seismic')


# In[49]:


#Dropping City 1 & City 2 as the correlation with the dependant variable is less than 0.1


# In[50]:


ds = ds.drop(['City1','City2'], axis = 1)


# In[51]:


ds.head()


# In[52]:


#Create scatter Plot of Independent Variable vs Dependent Variable.


# In[53]:


a = pd.plotting.scatter_matrix(ds, figsize = (15,20))


# In[54]:


#Treat “Average Fare” – 3rdColumn as your Dependent Variable and Rest of the columns as Independent Variable


# In[55]:


y = ds.pop('Average Fare')


# In[57]:


y = y.values.reshape(len(y),1)


# In[58]:


y.shape


# In[59]:


x = ds


# In[60]:


ds.head(6)


# In[61]:


x.columns


# In[ ]:


#Divide the data set into training and test data set and build a Multiple Linear Regression model.


# In[62]:


from sklearn.model_selection import train_test_split


# In[63]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3)


# In[64]:


from sklearn.linear_model import LinearRegression


# In[67]:


lr = LinearRegression()


# In[68]:


lr.fit(x_train,y_train)


# In[69]:


x_train.shape


# In[70]:


y_train.shape


# In[ ]:


#Print the accuracy of the overall model


# In[71]:


print(lr.score(x_train,y_train))
print(lr.score(x_test,y_test))


# In[ ]:


#Print the coefficients & intercepts of the linear regression model


# In[72]:


lr.coef_


# In[73]:


coef1 = pd.DataFrame(lr.coef_, columns = x_train.columns)


# In[74]:


x_train.columns


# In[75]:


coef1


# In[76]:


intercept = pd.DataFrame(lr.intercept_, columns = ["Intercept_Value"])


# In[77]:


intercept


# In[92]:


import statsmodels.api as sm
model = sm.OLS(y_train,x_train)
fitted1=model.fit()
#fitted1 =lr.fit(x_train,y_train)


# In[93]:


##Print the coefficients & intercepts of the linear regression model 
fitted1.summary()


# In[ ]:




