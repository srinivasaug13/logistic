#!/usr/bin/env python
# coding: utf-8

# In[27]:


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
from sklearn.model_selection import train_test_split


# In[28]:


df = pd.read_excel('low_30.xlsx')


# In[29]:


df.shape


# In[30]:


df.head(3)


# In[31]:


df.columns = df.columns.str.rstrip()


# In[32]:


df.info()


# In[33]:


df.columns


# In[124]:


df1 =df[['VCD','Viability','PCV final result','Offline pH','pO2','pCO2','GLN','GLUC','LAC','Ammonia','Sodium','Osmolality','Titer','Duration','Material_Batch']]


# In[125]:


sum_missing = df1.isnull().sum()
percent_missing = df1.isnull().sum() * 100 / len(df1)
missing_value_df = pd.DataFrame({'sum_missing':sum_missing,'percent_missing': percent_missing})
missing_value_df


# In[126]:


df.columns = df.columns.str.rstrip()


# In[127]:


df1.describe().transpose().round(3)


# In[128]:


df1.plot.box(grid = 'True')
plt.xticks(rotation=45)
plt.gcf().set_size_inches(15,8)


# In[129]:


df2=df1.drop('Material_Batch',axis=1)
rows = 3
cols = 4
#Creating subplot
fig, ax = plt.subplots(nrows = rows, ncols = cols, figsize = (16, 8))
#Iterating through each row and column of the testing dataframe
col = df2.columns
index = 0
for i in range(rows):
    for j in range(cols):
        sns.distplot(df2[col[index]], ax = ax[i][j])
        index += 1
plt.tight_layout()


# In[130]:


sns.set(style="ticks", color_codes=True)
sns.pairplot(df1, diag_kind = 'kde', hue = 'Titer')


# In[131]:


sns.set(color_codes=True)
df1.corr()
plt.subplots(figsize=(15,5))
sns.heatmap(df1.corr(), cmap= 'YlGnBu', annot=True)


# In[218]:


fig = px.imshow(df1.corr())
fig.update_layout(title='Correlation Matrix among X variables')
fig.show()


# In[132]:


plt.scatter(df1['Material_Batch'],df1['Titer'])
plt.xticks(rotation=90)
plt.gcf().set_size_inches(30,10)
plt.show()


# In[133]:


plt.scatter(df['Material_Batch'],df['VCD'])
plt.xticks(rotation=90)
plt.gcf().set_size_inches(40,10)
plt.show()


# In[134]:


fig,ax = plt.subplots()
plt.scatter(df1['VCD'],df1['Viability'])
ax.set_ylabel('Viability')
ax.set_xlabel('VCD')           
plt.xticks(rotation=90)
plt.gcf().set_size_inches(10,5)
plt.show()


# In[135]:


fig,ax = plt.subplots()
plt.scatter(df1['VCD'],df1['PCV final result'])
ax.set_ylabel('PCV final result')
ax.set_xlabel('VCD')           
plt.xticks(rotation=90)
plt.gcf().set_size_inches(10,5)
plt.show()


# In[136]:


fig,ax = plt.subplots()
plt.scatter(df1['PCV final result'],df1['Titer'])
ax.set_ylabel('Titer')
ax.set_xlabel('PCV final result')           
plt.xticks(rotation=90)
plt.gcf().set_size_inches(10,5)
plt.show()


# In[137]:


fig,ax = plt.subplots()
plt.scatter(df1['VCD'],df1['Offline pH'])
ax.set_ylabel('Offline pH')
ax.set_xlabel('VCD')           
plt.xticks(rotation=90)
plt.gcf().set_size_inches(10,5)
plt.show()


# In[138]:


fig,ax = plt.subplots()
plt.scatter(df1['PCV final result'],df1['Offline pH'])
ax.set_ylabel('Offline pH')
ax.set_xlabel('PCV final result')           
plt.xticks(rotation=90)
plt.gcf().set_size_inches(10,5)
plt.show()


# In[139]:


fig,ax = plt.subplots()
plt.scatter(df1['Duration'],df1['Offline pH'])
ax.set_ylabel('Offline pH')
ax.set_xlabel('Duration')           
plt.xticks(rotation=90)
plt.gcf().set_size_inches(10,5)
plt.show()


# In[140]:


fig,ax = plt.subplots()
plt.scatter(df1['Duration'],df1['PCV final result'])
ax.set_ylabel('PCV final result')
ax.set_xlabel('Duration')           
plt.xticks(rotation=90)
plt.gcf().set_size_inches(10,5)
plt.show()


# In[141]:


fig,ax = plt.subplots()
plt.scatter(df1['Duration'],df1['VCD'])
ax.set_ylabel('VCD')
ax.set_xlabel('Duration')           
plt.xticks(rotation=90)
plt.gcf().set_size_inches(10,5)
plt.show()


# In[142]:


fig,ax = plt.subplots()
plt.scatter(df1['Duration'],df1['Viability'])
ax.set_ylabel('Viability')
ax.set_xlabel('Duration')           
plt.xticks(rotation=90)
plt.gcf().set_size_inches(10,5)
plt.show()


# In[143]:


fig,ax = plt.subplots()
plt.scatter(df1['Duration'],df1['pCO2'])
ax.set_ylabel('pCO2')
ax.set_xlabel('Duration')           
plt.xticks(rotation=90)
plt.gcf().set_size_inches(10,5)
plt.show()


# In[144]:


sns.lineplot(x="Duration", y= "PCV final result", data=df,hue='Material_Batch',legend =False)
plt.xticks(rotation=90)
plt.gcf().set_size_inches(10,5)
plt.show()


# In[145]:


sns.lineplot(x="VCD", y= "PCV final result", data=df,hue='Material_Batch',legend =False)
plt.xticks(rotation=90)
plt.gcf().set_size_inches(10,5)
plt.show()


# In[146]:


sns.lineplot(x="VCD", y= "Viability", data=df,hue='Material_Batch',legend =False)
plt.xticks(rotation=90)
plt.gcf().set_size_inches(10,5)
plt.show()


# In[147]:


df1.shape


# In[148]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn import metrics


# In[162]:


X1= df1[['Duration','VCD','Viability','PCV final result','Offline pH','pO2','pCO2','GLN','GLUC','LAC','Ammonia','Sodium','Osmolality']]
y1= df1['Titer']
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, random_state= 124)


# In[163]:


X1.shape


# In[164]:


y1.shape


# In[165]:


X1.info()


# In[166]:


y1.info()


# In[167]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
pd.DataFrame({'variables':df2.columns[:-1], 'VIF':[variance_inflation_factor(df2.values, i+1) for i in range(len(df2.columns[:-1]))]})


# In[168]:


pivot1 = pd.pivot_table(data= df1, index='Material_Batch', values=['Material_Batch','Duration','VCD','Viability','PCV final result','Offline pH','pO2','pCO2','GLN','GLUC','LAC','Ammonia','Sodium','Osmolality'], aggfunc=['mean'])
pivot1.columns = pivot1.columns.droplevel(0)
pivot1 = pivot1.reset_index()
pivot1.round()


# In[169]:


sns.set(color_codes=True)
pivot1.corr()
plt.subplots(figsize=(15,5))
sns.heatmap(pivot1.corr(), cmap= 'YlGnBu', annot=True)


# In[170]:


plt.scatter(pivot1['Material_Batch'],pivot1['VCD'])
plt.xticks(rotation=90)
plt.gcf().set_size_inches(30,10)
plt.show()


# In[176]:


y1


# In[171]:


lr = LinearRegression().fit(X1_train,y1_train)
print(lr.intercept_)
print(lr.coef_)


# In[172]:


y1_pred = lr.predict(X1_test)


# In[173]:


y1_pred


# In[174]:


df_compare= pd.DataFrame({'Actual': y1_test, 'Predicted': y1_pred})
df_compare.head()


# In[175]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y1_test, y1_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y1_test, y1_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y1_test, y1_pred)))
print('R Square:', metrics.r2_score(y1_test, y1_pred))


# In[181]:


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


# In[182]:


# Fit Linear Model
datascaler = StandardScaler()
does = datascaler.fit_transform(X1.values)
X = sm.add_constant(does)
y = y1.values
mod = sm.OLS(y, X)
res = mod.fit()
print(res.summary())


# In[183]:


# Use data normalization
use_norm = True
# Use bias term
use_bias = True
# Use polynomial features of degree
use_degree = 1
# Use only interaction terms form higher degree polynomial features
use_inter = False


# In[184]:


# Define pipeline
pfeatures = PolynomialFeatures(degree=use_degree,interaction_only=use_inter, include_bias=use_bias)
pscaler = StandardScaler(with_mean=use_norm,with_std=use_norm)
lm = LinearRegression()
pipe = Pipeline([('features', pfeatures),('scaler', pscaler), ('model', lm)])
# Fit model
X = X1
y = y1
pipe.fit(X,y)


# In[185]:


# Model coefficients
columns = pfeatures.get_feature_names_out()
coefficients = lm.coef_
fig = px.bar(x=list(columns),y=coefficients.reshape(-1),title="Linear model coefficients",labels={'x':"Variables", 'y':"Estimated Coefficients"})
fig.show()


# In[186]:


# Plot predictions in train set
yhat = pipe.predict(X)
fig = px.scatter(x=y.values.reshape(-1),y=yhat.reshape(-1), title="Observed vs Predicted on Train Set; R^2 = "+str(round(pipe.score(X,y),3)), labels=dict(x="Observed Titer", y="Predicted Titer"))
fig.update_layout(shapes = [{'type': 'line', 'yref': 'paper', 'xref': 'paper', 'y0': 0, 'y1': 1, 'x0': 0, 'x1': 1, 'layer': 'below'}])
fig.show()
# Plot predictions in test set
X_test = X1_test
y_test = y1_test
yhat_test = pipe.predict(X_test)
fig = px.scatter(x=y_test.values.reshape(-1),y=yhat_test.reshape(-1), title="Observed vs Predicted on Test Set; R^2 = "+str(round(pipe.score(X_test,y_test),3)), labels=dict(x="Observed Titer", y="Predicted Titer"))
fig.update_layout(shapes = [{'type': 'line', 'yref': 'paper', 'xref': 'paper', 'y0': 0, 'y1': 1, 'x0': 0, 'x1': 1, 'layer': 'below'}])
fig.show()


# In[187]:


print("Error in test set prediction")
# Absolute RMSEP
rmse_abs = mean_squared_error(y_test, yhat_test,squared=False)
print('Absolute RMSEP: ',rmse_abs)
# Relative RMSEP
rmse_rel = mean_squared_error(y_test, yhat_test,squared=False) / np.std(np.array(y_test))
print('Relative RMSEP: ',rmse_rel)


# In[188]:


# Use data normalization
use_norm = True
# Number of latent variables
N_LV = 5


# In[189]:


# Define Pipeline
pscaler = StandardScaler(with_mean=use_norm,with_std=use_norm)
pls = PLSRegression(n_components=N_LV)
pipe = Pipeline([('scaler', pscaler), ('model', pls)])
# Train PLS model
X = X1
y = y1
pipe.fit(X,y)


# In[190]:


# Plot predictions in train set
yhat = pipe.predict(X)
fig = px.scatter(x=y.values.reshape(-1),y=yhat.reshape(-1), title="Observed vs Predicted on Train Set; R^2 = "+str(round(pipe.score(X,y),3)), labels=dict(x="Observed Titer", y="Predicted Titer"))
fig.update_layout(shapes = [{'type': 'line', 'yref': 'paper', 'xref': 'paper', 'y0': 0, 'y1': 1, 'x0': 0, 'x1': 1, 'layer': 'below'}])
fig.show()
# Plot predictions in test set
X_test = X1_test
y_test = y1_test
yhat_test = pipe.predict(X_test)
fig = px.scatter(x=y_test.values.reshape(-1),y=yhat_test.reshape(-1), title="Observed vs Predicted on Test Set; R^2 = "+str(round(pipe.score(X_test,y_test),3)), labels=dict(x="Observed Titer", y="Predicted Titer"))
fig.update_layout(shapes = [{'type': 'line', 'yref': 'paper', 'xref': 'paper', 'y0': 0, 'y1': 1, 'x0': 0, 'x1': 1, 'layer': 'below'}])
fig.show()


# In[191]:


# Absolute RMSEP
rmse_abs = mean_squared_error(y_test, yhat_test,squared=False)
print('Absolute RMSEP: ',rmse_abs)
# Relative RMSEP
rmse_rel = mean_squared_error(y_test, yhat_test,squared=False) / np.std(np.array(y_test))
print('Relative RMSEP: ',rmse_rel)


# In[192]:


# Plot Explained Variance plots
scores = pls.x_scores_
expl_var = np.var(scores, axis=0)
expl_var_ratio = expl_var / np.sum(expl_var)
fig = px.line(x=list(range(N_LV)), y=np.cumsum(expl_var_ratio), color=px.Constant("Cumulative explained variance"), labels=dict(x="Principal component index", y="Explained Variance Ratio", color="Legend"))
fig.add_bar(x=list(range(N_LV)), y=expl_var_ratio, name="Individual explained variance")
fig.show()


# In[193]:


# Principal component on x-axis
select_x_pca = 1
# Principal component on y-axis
select_y_pca = 2


# In[195]:


# Plot loadings
loadings = pls.x_loadings_
fig = px.scatter(x=pls.x_scores_[:,select_x_pca-1],y=pls.x_scores_[:,select_y_pca-1], title="PLS Score and loadings plot",labels={'x':"PC - "+ str(select_x_pca) , 'y':"PC -"+ str(select_y_pca)})
for i, feature in enumerate(X1.columns):
    fig.add_shape(type='line', x0=0, y0=0, x1=loadings[i,select_x_pca-1], y1=loadings[i,select_y_pca-1])
    fig.add_annotation(x=loadings[i,select_x_pca-1], y=loadings[i,select_y_pca-1], ax=0, ay=0, xanchor="center", yanchor="bottom", text=feature)
fig.show()


# In[208]:


def vip(x, y, model):
    import numpy as np
    t = model.x_scores_
    w = model.x_weights_
    q = model.y_loadings_

    m, p = x.shape
    _, h = t.shape

    vips = np.zeros((p,))

    s = np.diag(t.T @ t @ q.T @ q).reshape(h, -1)
    total_s = np.sum(s)

    for i in range(p):
        weight = np.array([ (w[i,j] / np.linalg.norm(w[:,j]))**2 for j in range(h) ])
        vips[i] = np.sqrt(p*(s.T @ weight)/total_s)

    return vips


# In[209]:


# Model coefficients
columns = X.columns
coefficients = vip(X,y,pls)
fig = px.bar(x=list(columns),y=coefficients.reshape(-1),title="Variable Importance in PLS", labels={'x':"Variables", 'y':"VIP values"})
fig.show()


# In[214]:


# Range of number of latent variables to optimize for.
range_LV = range(1, 6)
# Use k Fold for Cross Validation
use_folds = 10


# In[215]:


# Define Pipeline
pscaler = StandardScaler(with_mean=use_norm,with_std=use_norm)
pls = PLSRegression(n_components=N_LV)
pipe = Pipeline([('scaler', pscaler), ('pls', pls)])
# Train PLS model
pipe.fit(X,y)
# Obtain cross validation curve
train_eval, valid_eval = validation_curve(pipe, X, y, param_name = "pls__n_components", param_range=list(range_LV),scoring = 'neg_root_mean_squared_error')


# In[216]:


train_score=-np.mean(train_eval,axis=1)
valid_score=-np.mean(valid_eval,axis=1)
train_std = np.std(train_eval,axis=1)
valid_std = np.std(valid_eval,axis=1)


# In[217]:


fig = go.Figure()
fig.add_trace(go.Scatter( x=list(range_LV), y=train_score, error_y=dict(type='data',array=train_std,visible=True), name="Training"))
fig.add_trace(go.Scatter( x=list(range_LV), y=valid_score, error_y=dict(type='data',array=valid_std,visible=True), name="Validation"))
fig.update_layout(title="Hyperparameter Optimization in PLS",xaxis_title="Number of Latent Variables",yaxis_title="RMSE",legend_title="Evaluation type")
fig.show()


# In[ ]:




