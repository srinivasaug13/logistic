#!/usr/bin/env python
# coding: utf-8

# In[126]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[127]:


bf = pd.read_csv("Bank_Personal_Loan_Modelling.csv")


# In[129]:


bf.head(6)


# In[130]:


bf.shape


# In[131]:


bf.describe().T


# In[133]:


# checking missing values
print (bf.isnull().sum())


# In[46]:


## Target Variable Frequency Distribution
freq = bf['Personal Loan'].value_counts().to_frame()
freq.reset_index(inplace=True)
freq.columns = [freq.columns[1], 'count']
freq['prop'] = freq['count'] / sum(freq['count'])
print (freq)

import seaborn as sns
sns.countplot(Bank_DF['Personal Loan'])


# In[47]:


sns.pairplot(bf)


# In[48]:


sns.distplot(Bank_DF["Age"],bins = "auto")
### mean and median are similar for "age" variable. hence distribution of varibale is normal.


# In[49]:


sns.boxplot(y = 'Age', data = Bank_DF)


# In[50]:


sns.distplot(Bank_DF["Experience"],bins = "auto")
### mean and median are similar for "Experience" variable. hence distribution of varibale is normal.


# In[51]:


sns.boxplot(y = 'Experience', data = Bank_DF)


# In[52]:


sns.distplot(Bank_DF["Income"],bins = "auto")
# mean is greater than median. "Income" variable distribution is right skewed.
# box plot of "Income variable" shows outliers and mean is getting affected by outliers.


# In[53]:



sns.boxplot(y = 'Income', data = Bank_DF)


# In[54]:


sns.distplot(Bank_DF["CCAvg"],bins = "auto")
# mean is greater than median. "CCAvg" variable distribution is right skewed.
# box plot of "CCAvg variable" shows outliers and mean is getting affected by outliers.


# In[55]:



sns.boxplot(y = 'CCAvg', data = Bank_DF)


# In[56]:



sns.distplot(Bank_DF["Mortgage"],bins = "auto")


# In[57]:


sns.boxplot(y = 'Mortgage', data = Bank_DF)


# In[58]:


freq = Bank_DF['Family'].value_counts().to_frame()
freq.reset_index(inplace=True)
freq.columns = [freq.columns[1], 'count']
freq['prop'] = freq['count'] / sum(freq['count'])
print (freq)

import seaborn as sns
sns.countplot(Bank_DF['Family'])


# In[59]:


freq = Bank_DF['Education'].value_counts().to_frame()
freq.reset_index(inplace=True)
freq.columns = [freq.columns[1], 'count']
freq['prop'] = freq['count'] / sum(freq['count'])
print (freq)

import seaborn as sns
sns.countplot(Bank_DF['Education'])


# In[60]:


freq = Bank_DF['Securities Account'].value_counts().to_frame()
freq.reset_index(inplace=True)
freq.columns = [freq.columns[1], 'count']
freq['prop'] = freq['count'] / sum(freq['count'])
print (freq)

import seaborn as sns
sns.countplot(Bank_DF['Securities Account'])


# In[61]:


freq = Bank_DF['CD Account'].value_counts().to_frame()
freq.reset_index(inplace=True)
freq.columns = [freq.columns[1], 'count']
freq['prop'] = freq['count'] / sum(freq['count'])
print (freq)

import seaborn as sns
sns.countplot(Bank_DF['CD Account'])


# In[62]:


freq = Bank_DF['Online'].value_counts().to_frame()
freq.reset_index(inplace=True)
freq.columns = [freq.columns[1], 'count']
freq['prop'] = freq['count'] / sum(freq['count'])
print (freq)

import seaborn as sns
sns.countplot(Bank_DF['Online'])


# In[63]:


freq = Bank_DF['CreditCard'].value_counts().to_frame()
freq.reset_index(inplace=True)
freq.columns = [freq.columns[1], 'count']
freq['prop'] = freq['count'] / sum(freq['count'])
print (freq)

import seaborn as sns
sns.countplot(Bank_DF['CreditCard'])


# In[65]:


from sklearn.model_selection import train_test_split
from scipy.stats import zscore
independent= Bank_DF.drop(["ID", "ZIP Code","Personal Loan"],axis=1)
dependent= Bank_DF["Personal Loan"]
independent = independent.apply(zscore)
X_train, X_test, y_train, y_test = train_test_split(independent, dependent, test_size=0.3,random_state=5)


# In[67]:


from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
k_range = list(range(1, 31))
param_grid = dict(n_neighbors=k_range)
grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy', return_train_score=False)


# In[68]:



grid.fit(X_train,y_train)


# In[69]:


# view the results as a pandas DataFrame
pd.DataFrame(grid.cv_results_)[['mean_test_score', 'std_test_score', 'params']]


# In[70]:


# examine the first result
print(grid.cv_results_['params'][0])
print(grid.cv_results_['mean_test_score'][0])


# In[71]:


# print the array of mean scores only
grid_mean_scores = grid.cv_results_['mean_test_score']
print(grid_mean_scores)


# In[72]:


# plot the results
plt.plot(k_range, grid_mean_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')


# In[73]:


# examine the best model
print(grid.best_score_)
print(grid.best_params_)
print(grid.best_estimator_)


# In[74]:


KNN = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_jobs=1, n_neighbors=3, p=2,
           weights='uniform')


# In[75]:


KNN.fit(X_train,y_train)


# In[76]:


y_pred = KNN.predict(X_test)


# In[79]:


from sklearn.metrics import confusion_matrix
# save confusion matrix and slice into four pieces
confusion = confusion_matrix(y_test, y_pred)
print(confusion)
#[row, column]
TP = confusion[1, 1]
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]


# In[82]:


# Accuracy calculation
from sklearn.metrics import accuracy_score
print((TP + TN) / float(TP + TN + FP + FN))
print(accuracy_score(y_test, y_pred))


# In[83]:


## classification error
classification_error = (FP + FN) / float(TP + TN + FP + FN)
print(classification_error)
print(1 - accuracy_score(y_test, y_pred))


# In[87]:


from sklearn.metrics import recall_score
sensitivity = TP / float(FN + TP)
print(sensitivity)
print(recall_score(y_test, y_pred))


# In[88]:


specificity = TN / (TN + FP)

print(specificity)


# In[89]:


false_positive_rate = FP / float(TN + FP)
print(false_positive_rate)
print(1 - specificity)


# In[91]:


from sklearn.metrics import precision_score
precision = TP / float(TP + FP)
print(precision)
print(precision_score(y_test, y_pred))


# In[92]:


from sklearn.metrics import roc_curve, auc
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(false_positive_rate, true_positive_rate)
roc_auc


# In[93]:


y_pred_proba = KNN.predict_proba(X_test)[:,1]


# In[94]:


from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.plot([0,1],[0,1],'k--')
plt.plot(fpr,tpr, label='Knn')
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.title('Knn(n_neighbors=3) ROC curve')
plt.show()


# In[95]:


#Area under ROC curve
from sklearn.metrics import roc_auc_score
roc_auc_score(y_test,y_pred_proba)


# In[96]:


# calculate cross-validated AUC
from sklearn.model_selection import cross_val_score
cross_val_score(KNN, X_train, y_train, cv=10, scoring='roc_auc').mean()


# In[97]:


# train a logistic regression model on the training set
from sklearn.linear_model import LogisticRegression

# instantiate model
logreg = LogisticRegression()

# fit model
logreg.fit(X_train, y_train)


# In[98]:


# make class predictions for the testing set
y_pred_class = logreg.predict(X_test)


# In[100]:


# save confusion matrix and slice into four pieces
confusion = confusion_matrix(y_test, y_pred_class)
print(confusion)
#[row, column]
TP = confusion[1, 1]
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]


# In[101]:


# calculate accuracy
from sklearn import metrics
print(metrics.accuracy_score(y_test, y_pred_class))


# In[102]:


## classification error
classification_error = (FP + FN) / float(TP + TN + FP + FN)
print(classification_error)
print(1 - metrics.accuracy_score(y_test, y_pred_class))


# In[103]:


sensitivity = TP / float(FN + TP)

print(sensitivity)
print(metrics.recall_score(y_test, y_pred_class))


# In[104]:


specificity = TN / (TN + FP)

print(specificity)


# In[105]:


false_positive_rate = FP / float(TN + FP)
print(false_positive_rate)
print(1 - specificity)


# In[106]:


precision = TP / float(TP + FP)

print(precision)
print(metrics.precision_score(y_test,y_pred_class))


# In[107]:


from sklearn.metrics import roc_curve, auc
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test,y_pred_class)
roc_auc = auc(false_positive_rate, true_positive_rate)
roc_auc


# In[108]:


y_pred_proba_1 = logreg.predict_proba(X_test)[:,1]


# In[109]:


from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba_1)
plt.plot([0,1],[0,1],'k--')
plt.plot(fpr,tpr, label='Logistic Regression')
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.title('Logistic Regression ROC curve')
plt.show()


# In[110]:


#Area under ROC curve
from sklearn.metrics import roc_auc_score
roc_auc_score(y_test,y_pred_proba_1)


# In[111]:


# calculate cross-validated AUC
from sklearn.model_selection import cross_val_score
cross_val_score(logreg, X_train, y_train, cv=10, scoring='roc_auc').mean()


# In[112]:


from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
NB = BernoulliNB()
NB.fit(X_train, y_train)


# In[113]:


# make class predictions for the testing set
y_pred_class1 = NB.predict(X_test)


# In[114]:


# save confusion matrix and slice into four pieces
from sklearn import metrics
confusion = metrics.confusion_matrix(y_test, y_pred_class1)
print(confusion)
#[row, column]
TP = confusion[1, 1]
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]


# In[115]:


# calculate accuracy
from sklearn import metrics
print(metrics.accuracy_score(y_test, y_pred_class))


# In[116]:


## classification error
classification_error = (FP + FN) / float(TP + TN + FP + FN)
print(classification_error)
print(1 - metrics.accuracy_score(y_test, y_pred_class1))


# In[117]:


sensitivity = TP / float(FN + TP)

print(sensitivity)
print(metrics.recall_score(y_test, y_pred_class1))


# In[118]:


specificity = TN / (TN + FP)

print(specificity)


# In[119]:


false_positive_rate = FP / float(TN + FP)
print(false_positive_rate)
print(1 - specificity)


# In[120]:


precision = TP / float(TP + FP)

print(precision)
print(metrics.precision_score(y_test,y_pred_class1))


# In[121]:


from sklearn.metrics import roc_curve, auc
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test,y_pred_class1)
roc_auc = auc(false_positive_rate, true_positive_rate)
roc_auc


# In[122]:


y_pred_proba_2 = NB.predict_proba(X_test)[:,1]


# In[123]:


from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba_2)
plt.plot([0,1],[0,1],'k--')
plt.plot(fpr,tpr, label='Naive Bayes')
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.title('Naive Bayes ROC curve')
plt.show()


# In[124]:


#Area under ROC curve
from sklearn.metrics import roc_auc_score
roc_auc_score(y_test,y_pred_proba_2)


# In[125]:


# calculate cross-validated AUC
from sklearn.model_selection import cross_val_score
cross_val_score(NB, X_train, y_train, cv=10, scoring='roc_auc').mean()


# # CONCLUSION
# Logistic Regression model performs better than KNN and naive bayes models in terms of metrics Accuracy, sensitivity, specificity and CV ROC_AUC.

# In[ ]:




