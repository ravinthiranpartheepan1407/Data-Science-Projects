#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


data = "F:/Applied Informatics/Semester-III/Data Visualization/Laboratory Works/Homework 6 and 7/Dataset/cars.csv"


# In[5]:


data


# In[13]:


dataf = pd.read_csv(data, names=['mpg','cylinders','cubicinches','hp','weightlbs','time-to-60','year','brand'])


# In[14]:


dataf.head()


# In[64]:


features = ['mpg','cylinders','cubicinches','hp']
cars = dataf.loc[2:30, features].values


# In[65]:


cars_y = dataf.loc[2:30,['brand']].values


# In[66]:


carss = StandardScaler().fit_transform(cars)


# In[87]:


pd.DataFrame(data = cars, columns = features).head(30)


# In[68]:


pca = PCA(n_components = 2)


# In[69]:


principalComponents = pca.fit_transform(cars)


# In[70]:


principalDf = pd.DataFrame(data = principalComponents, columns = ['PrincipalComponent_1','PrincipalComponent_2'])


# In[85]:


principalDf.head(30)


# In[86]:


dataf[['brand']].head(30)


# In[74]:


finalDf = pd.concat([principalDf, dataf[['brand']]], axis = 1)


# In[88]:


finalDf.head(30)


# In[97]:


fig = plt.figure(figsize = (15,15))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('PCA for Car Dataset', fontsize = 20)


targets = ['Europe', 'US.','Japan']
colors = ['r', 'g', 'b']
for brand, color in zip(targets,colors):
    indicesToKeep = finalDf['brand'] == brand
    ax.scatter(finalDf.loc[indicesToKeep, 'PrincipalComponent_1']
               , finalDf.loc[indicesToKeep, 'PrincipalComponent_2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()


# In[ ]:




