#!/usr/bin/env python
# coding: utf-8

# In[1]:


import time  # to simulate a real time data, time loop

import numpy as np  # np mean, np random
import pandas as pd  # read csv, df manipulation
import plotly.express as px  # interactive charts
import streamlit as st  # 🎈 data web app development


# In[2]:


st.set_page_config(
    page_title="Real-Time process Yield Monitor",
    page_icon="✅",
    layout="wide",
)


# In[3]:


df = pd.read_csv("process Monitor.csv")


# In[4]:


# dashboard title
st.title("Real-Time / Live Process Monitor")


# In[5]:


# top-level filters
Batch_filter = st.selectbox("Select the Batch", pd.unique(df["Batch"]))


# In[6]:


# creating a single-element container
placeholder = st.empty()


# In[7]:


# dataframe filter
df = df[df["Batch"] == Batch_filter]


# In[8]:


# creating KPIs
avg_Glc = np.mean(df["Glc"])
avg_Gln = np.mean(df["Gln"])
avg_LAC = np.mean(df["LAC"])
avg_VCD_xE5 = np.mean(df["VCD_xE5"])
avg_VCD = np.mean(df["VCD"])
count_Days = int(
   df[(df["Days"] == "Days")]["Days"].count()
   + np.random.choice(range(0, 12))
   )
Titer = np.mean(df["Titer"])


# In[9]:


import streamlit as st


# Assuming avg_Glc, avg_Gln, avg_LAC, VCD_xE5, VCD, count_Days are defined elsewhere

# create seven columns
kpi1, kpi2, kpi3, kpi4, kpi5, kpi6  = st.columns(6)

# fill in those three columns with respective metrics or KPIs
kpi1.metric(
    label = "Glc",
    value=round(avg_Glc),
    delta=round(avg_Glc) - 1,
)

kpi2.metric(
    label = "Gln",
    value=round(avg_Gln),
    delta=round(avg_Gln) - 1,
)

kpi3.metric(
    label = "LAC",
    value=round(avg_LAC),
    delta=round(avg_LAC) - 1,
)

kpi5.metric(
    label="VCD",
    value=int(avg_VCD),
    delta=-1 + avg_VCD,
)

kpi6.metric(
    label="Days",
    value=int(count_Days),
    delta=-1 + count_Days,
)


# In[10]:


fig_col1, fig_col2, fig_col3, fig_col4, fig_col5 = st.columns(5)

with fig_col1:
    st.markdown("### First Chart")
    fig = px.scatter(df, x="Days", y="VCD", title='VCD vs Duration(days)', color='Batch')
    st.write(fig)

with fig_col2:
    st.markdown("### Second Chart")
    fig2 = px.histogram(data_frame=df, x="Glc")
    st.write(fig2)

with fig_col3:
    st.markdown("### Third Chart")
    fig3 = px.ecdf(data_frame=df, x="Gln")
    st.write(fig3)

with fig_col4:
    st.markdown("### Fourth Chart")
    fig4 = px.histogram(data_frame=df, x="LAC")
    st.write(fig4)

with fig_col5:
    st.markdown("### Fifth Chart")
    fig5 = px.histogram(data_frame=df, x="VCD")
    st.write(fig5)

st.markdown("### Detailed Data View")
st.dataframe(df)


# In[ ]:




