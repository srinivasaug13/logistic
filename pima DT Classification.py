#!/usr/bin/env python
# coding: utf-8

# In[7]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import metrics
import warnings
warnings.filterwarnings('ignore')
import random
import matplotlib.pyplot as plt


# In[8]:


#colnames=['preg','plas','pres','skin','test','mass','pedi','age','class']
pima_df=pd.read_csv("pima-indians-diabetes.csv")


# In[9]:


pima_df.head(10)
#0s signify a lot of missing values


# In[10]:


pima_df.shape


# In[11]:


pima_df.dtypes


# In[12]:


pima_df.info()


# In[13]:


print(pima_df.describe())


# In[14]:


print((pima_df[['preg','plas','pres','skin','test','mass','pedi','age']] == 0).sum())


# In[15]:


pima_df.describe().transpose()


# In[16]:


print(pima_df.isnull().sum())


# In[17]:


pima_df[~pima_df.applymap(np.isreal).all(1)]


# In[18]:


sns.heatmap(pima_df.isnull(), cbar=False)


# In[19]:


import seaborn as sns
sns.pairplot(pima_df, hue="class", palette="husl")


# In[20]:


#pima_df=pima_df.fillna(pima_df.median())


# In[21]:


pima_df.loc[pima_df.plas == 0, 'plas'] = pima_df.plas.median()
pima_df.loc[pima_df.pres == 0, 'pres'] = pima_df.pres.median()
pima_df.loc[pima_df.skin == 0, 'skin'] = pima_df.skin.median()
pima_df.loc[pima_df.test == 0, 'test'] = pima_df.test.median()
pima_df.loc[pima_df.mass == 0, 'mass'] = pima_df.mass.median()


# In[22]:


pima_df.describe().transpose()


# In[23]:


pima_df.groupby("class").agg({'class': 'count'})


# In[24]:


pima_df.groupby(["class"]).count()


# In[26]:


import seaborn as sns
sns.pairplot(pima_df, hue="class", palette="husl")
sns.pairplot(pima_df,hue="class")


# In[27]:


import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
sns.heatmap(pima_df.corr(), annot=True, vmin=-1, vmax=1, cmap='seismic')


# In[28]:


pima_df.corr()


# In[29]:


## Target Variable Frequency Distribution
freq = pima_df['class'].value_counts().to_frame()
freq.reset_index(inplace=True)
freq.columns = [freq.columns[1], 'count']
freq['prop'] = freq['count'] / sum(freq['count'])
print (freq)
import seaborn as sns
sns.countplot(pima_df['class'])


# In[30]:


array= pima_df.values
X=array[:,0:7]
Y=array[:,8]
test_size=0.30
seed=7
X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.30,random_state=seed)


# In[59]:


#X_train = pima_df.head(538)
#X_test = pima_df.tail(230)

#y_train = X_train.pop("class")
#y_test = X_test.pop("class")


# In[31]:


from sklearn.tree import DecisionTreeClassifier
dt_model = DecisionTreeClassifier(criterion = 'entropy' )
dt_model.fit(X_train, y_train)


# In[32]:


# splitting data into training and test set for independent attributes
#n=pima_df['class'].count()
#train_set = pima_df.head(int(round(n*0.7))) # Up to the last initial training set row
#test_set = pima_df.tail(int(round(n*0.3))) # Past the last initial training set row

# capture the target column ("class") into separate vectors for training set and test set
#train_labels = train_set.pop("class")
#test_labels = test_set.pop("class")


# In[33]:


#from sklearn.tree import DecisionTreeClassifier
#dt_model = DecisionTreeClassifier(criterion = 'entropy' )
#dt_model.fit(train_set, train_labels)


# In[35]:


#test_pred = dt_model.predict(y_test)
dt_model.score(X_test , y_test)


# In[36]:


print(dt_model.score(X_train , y_train))
print(dt_model.score(X_test , y_test))


# In[60]:


from IPython.display import image
#import pydotplus as pydot
from sklearn import tree
from os import system

train_char_label = ['No','Yes']
Credit_Tree_File = open('d:\pima_tree.dot','w')
dot_data = tree.export_grapviz(dt_model,out_file=Credit_Tree_File,feature_names=list(X_train),class_names = list(train_char_label))
Credit_Tree_File.close()
print (pd.DataFrame(dt_model.feature_importances_, columns = ["Imp"], index = train_set.columns))


# In[43]:


print (pd.DataFrame(dt_model.feature_importances_, columns = ["Imp"], index = X_train.columns))


# In[61]:


system("dot -Tpng D:\pima_tree.dot -o D:/pima_tree.png")
image("d:\pima_tree.png")


# In[62]:


y_predict=dt_model.predict(X_test)


# In[63]:


print(metrics.confusion_matrix(y_test, y_predict))


# improve the model

# In[76]:


from sklearn.tree import DecisionTreeClassifier
#dt_model = DecisionTreeClassifier(criterion = 'entropy',class_weight={0:.5,1:.5},max_depth=5, min_samples_leaf=5)
dt_model = DecisionTreeClassifier(criterion = 'entropy',max_depth=3)
dt_model.fit(X_train, y_train)


# In[77]:


#from IPython.display import image
from sklearn import tree
from os import system
Credit_Tree_File = open('d:\pima_tree_regularized.dot','w')
dot_data = tree.export_grapviz(dt_model,out_file=Credit_Tree_File,feature_names=list(X_train),class_names = list(train_char_label))
Credit_Tree_File.close()

system("dot -Tpng D:\pima_tree_regularized.dot -o D:/pima_tree_regularized.png")
image("d:\pima_tree_regularized.png")


# In[78]:


print(dt_model.score(X_train, y_train))
print(dt_model.score(X_test, y_test))


# In[79]:


y_predict = dt_model.predict(X_test)


# In[80]:


print(metrics.confusion_matrix(y_test,y_predict))


# In[ ]:





# In[ ]:




