#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import libaries
import pandas as pd
import numpy as np
import scipy
import importlib  
import scipy.integrate
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import statsmodels.api as sm
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GridSearchCV, validation_curve
from sklearn.metrics import mean_squared_error


# In[2]:


import numpy as np
import pandas as pd
from scipy.stats import zscore
from sklearn.metrics import accuracy_score
import seaborn as sns
import os
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.metrics import mean_squared_error as mse
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import validation_curve
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import r2_score
#import missingno as msno
import warnings 
warnings.filterwarnings('ignore')


# In[3]:


titer= pd.read_csv("titer1.csv", index_col=0)


# In[4]:


titer.head(6)


# In[5]:


titer.info()


# In[6]:


titer.dtypes


# In[7]:


titer.shape


# In[8]:


titer.describe().transpose()


# In[9]:


print(titer.isnull().any())


# In[10]:


print(titer.isnull().sum())


# In[11]:


#import missingno as msno
sns.heatmap(titer.isnull(), cbar=False)


# In[12]:


#sns.pairplot(titer)


# In[13]:


sns.distplot(titer['VCD'],bins ="auto")


# In[14]:


sns.distplot(titer['Titer'],bins ="auto")


# In[15]:


sns.distplot(titer['Glc'],bins ="auto")


# In[16]:


sns.distplot(titer['Gln'],bins ="auto")


# In[17]:


sns.distplot(titer['Lac'],bins ="auto")


# In[18]:


sns.boxplot(y="Titer", data=titer)
plt.figure(figsize=(100,5))


# In[19]:


sns.boxplot(y="Lac", data=titer)
plt.figure(figsize=(100,5))


# In[20]:


sns.boxplot(y="VCD", data=titer)
plt.figure(figsize=(100,5))


# In[21]:


sns.boxplot(y="Glc", data=titer)
plt.figure(figsize=(100,5))


# In[22]:


sns.boxplot(y="Gln", data=titer)
plt.figure(figsize=(100,5))


# In[ ]:





# In[ ]:


sns.pairplot(titer[['Glc','Titer','Lac','Gln', 'VCD']], diag_kind= "kde")


# In[ ]:


#Remove the outliers from the data
titer.head(20)
Q1 = titer.quantile(0.25)
Q3 = titer.quantile(0.75)
IQR = Q3 - Q1
titer_out = titer[~((titer < (Q1 - 1.5 * IQR)) |(titer > (Q3 + 1.5 * IQR))).any(axis=1)]
titer_out.head(4)


# In[ ]:


## Correlation Output
sns.heatmap(titer_out.corr(), annot=True, vmin=-1, vmax=1, cmap='seismic')


# In[ ]:


titer_out.describe().transpose()


# In[ ]:


## Transforming VCD Variable
titer_out["VCD"]=titer_out["VCD"].map(lambda
         x: -0.1-(x-0.1) if x<-0.1 else x)


# In[ ]:


sns.distplot(titer_out['VCD'],bins="auto")


# In[ ]:


titer_out["Titer"]=titer_out["Titer"].map(lambda
         x: -0.01-(x-0.01) if x<-0.01 else x)


# In[ ]:


sns.distplot(titer_out['Titer'],bins="auto")


# In[ ]:




