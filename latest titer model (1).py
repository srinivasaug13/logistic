#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set(color_codes=True)
import warnings
warnings.filterwarnings('ignore')
warnings.warn('DelftStack')
warnings.warn('Do not show this message')
print("No Warning Shown")


# In[4]:


titer= pd.read_csv("titer1.csv")


# In[5]:


titer.head(6)


# In[6]:


titer.info()


# In[7]:


titer.shape


# In[8]:


sum_missing = titer.isnull().sum()
percent_missing = titer.isnull().sum() * 100 / len(titer)
missing_value_df = pd.DataFrame({'sum_missing':sum_missing,'percent_missing': percent_missing})
missing_value_df 


# In[9]:


Numerical_variables = titer[['Glc','Titer','Lac','Gln', 'VCD']]


# In[13]:


Numerical_variables.describe().transpose()


# In[14]:


Numerical_variables.plot.box(grid = 'True')
plt.xticks(rotation=45)


# In[17]:


import warnings
warnings.filterwarnings('ignore')
warnings.warn('DelftStack')
warnings.warn('Do not show this message')
print("No Warning Shown")


# In[18]:


rows = 2
cols = 2
#Creating subplot
fig, ax = plt.subplots(nrows = rows, ncols = cols, figsize = (16, 8))
#Iterating through each row and column of the testing dataframe
col = Numerical_variables.columns
index = 0
for i in range(rows):
    for j in range(cols):
        sns.distplot(Numerical_variables[col[index]], ax = ax[i][j])
        index += 1
plt.tight_layout()


# In[19]:


sns.distplot(titer['VCD'],bins ="auto")


# In[20]:


sns.pairplot(Numerical_variables,diag_kind = 'kde')


# In[21]:


sns.set(color_codes=True)
Numerical_variables.corr()
sns.heatmap(Numerical_variables.corr(), cmap= 'YlGnBu', annot=True)


# In[22]:


titer.columns


# In[31]:


pivot1 = pd.pivot_table(data= titer, index='Batch', values=['Glc','Gln','Lac','VCD','Titer'], aggfunc=['mean'])
pivot1.columns = pivot1.columns.droplevel(0)
pivot1 = pivot1.reset_index()
pivot1.round()


# In[43]:


plt.scatter(pivot1['Batch'],pivot1['VCD'])
plt.xticks(rotation=90)
plt.show()
plt.figure(figsize = (15,15))


# In[44]:


plt.scatter(pivot1['Batch'],pivot1['Titer'])
plt.xticks(rotation=90)
plt.show()
plt.figure(figsize = (15,15))


# In[45]:


plt.scatter(pivot1['Batch'],pivot1['Glc'])
plt.xticks(rotation=90)
plt.show()
plt.figure(figsize = (15,15))


# In[46]:


plt.scatter(pivot1['Batch'],pivot1['Gln'])
plt.xticks(rotation=90)
plt.show()
plt.figure(figsize = (15,15))


# In[47]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn import metrics


# In[55]:


X1= titer[['Glc', 'Gln', 'Lac']]
y1= titer['VCD']
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, random_state= 124)


# In[56]:


lr = LinearRegression().fit(X1_train,y1_train)
print(lr.intercept_)
print(lr.coef_)


# In[57]:


y1_pred = lr.predict(X1_test)


# In[58]:


y1_pred


# In[59]:


df_compare= pd.DataFrame({'Actual': y1_test, 'Predicted': y1_pred})
df_compare.head()


# In[60]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y1_test, y1_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y1_test, y1_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y1_test, y1_pred)))
print('R Square:', metrics.r2_score(y1_test, y1_pred))


# In[ ]:





# In[ ]:




