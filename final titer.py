#!/usr/bin/env python
# coding: utf-8

# In[20]:


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


# In[21]:


titer= pd.read_csv("ravi_titer.csv")


# In[22]:


titer.head(50)


# In[ ]:


#initaial condition


# In[23]:


titer['VCD'] = titer['VCD'].apply(lambda x : x if x > 0 else 0)
titer['Titer'] = titer['Titer'].apply(lambda x : x if x > 0 else 0)


# In[24]:


titer['Titer']>0


# In[25]:


titer.info()


# In[26]:


titer.shape


# In[27]:


sum_missing = titer.isnull().sum()
percent_missing = titer.isnull().sum() * 100 / len(titer)
missing_value_df = pd.DataFrame({'sum_missing':sum_missing,'percent_missing': percent_missing})
missing_value_df 


# In[30]:


Numerical_variables = titer[['Glc','Titer','LAC','Gln', 'VCD']]


# In[31]:


Numerical_variables.describe().transpose()


# In[32]:


Numerical_variables.plot.box(grid = 'True')
plt.xticks(rotation=45)


# In[33]:


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


# In[34]:


sns.distplot(titer['VCD'],bins ="auto")


# In[35]:


sns.pairplot(Numerical_variables,diag_kind = 'kde')


# In[36]:


sns.set(color_codes=True)
Numerical_variables.corr()
sns.heatmap(Numerical_variables.corr(), cmap= 'YlGnBu', annot=True)


# In[37]:


titer.columns


# In[38]:


pivot1 = pd.pivot_table(data= titer, index='Batch', values=['Glc','Gln','LAC','VCD','Titer'], aggfunc=['mean'])
pivot1.columns = pivot1.columns.droplevel(0)
pivot1 = pivot1.reset_index()
pivot1.round()


# In[39]:


plt.scatter(pivot1['Batch'],pivot1['VCD'])
plt.xticks(rotation=90)
plt.show()
plt.figure(figsize = (15,15))


# In[40]:


plt.scatter(pivot1['Batch'],pivot1['Titer'])
plt.xticks(rotation=90)
plt.show()
plt.figure(figsize = (15,15))


# In[41]:


plt.scatter(pivot1['Batch'],pivot1['Glc'])
plt.xticks(rotation=90)
plt.show()
plt.figure(figsize = (15,15))


# In[42]:


plt.scatter(pivot1['Batch'],pivot1['Gln'])
plt.xticks(rotation=90)
plt.show()
plt.figure(figsize = (15,15))


# In[48]:


Numerical_variables1=titer[['Glc','Titer','LAC','Gln', 'VCD','Elapsed time']]


# In[49]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
pd.DataFrame({'variables':Numerical_variables1.columns[:-1], 'VIF':[variance_inflation_factor(Numerical_variables1.values, i+1) for i in range(len(Numerical_variables1.columns[:-1]))]})


# In[56]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn import metrics


# In[69]:


X1= titer[['Glc', 'Gln','LAC']]
y1= titer['Titer']
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, random_state= 124)


# In[70]:


lr = LinearRegression().fit(X1_train,y1_train)
print(lr.intercept_)
print(lr.coef_)


# In[71]:


y1_pred = lr.predict(X1_test)


# In[72]:


y1_pred


# In[73]:


df_compare= pd.DataFrame({'Actual': y1_test, 'Predicted': y1_pred})
df_compare.head()


# In[74]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y1_test, y1_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y1_test, y1_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y1_test, y1_pred)))
print('R Square:', metrics.r2_score(y1_test, y1_pred))


# In[ ]:


YEAR	MONTH	DAY	HRS	MIN


# In[ ]:




