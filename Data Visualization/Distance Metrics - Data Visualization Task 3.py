#!/usr/bin/env python
# coding: utf-8

# In[3]:


from pandas import Series


# In[7]:


data = Series.from_csv('F:/Applied Informatics/Semester-III/Data Visualization/Laboratory Works/Homework 3/archive/DailyClimateTest.csv', header=0)


# In[57]:


testing = data.head(10)


# In[212]:


tested = data.tail


# In[10]:


type(data)


# In[26]:


test = data['2017-03']


# In[16]:


data.size


# In[17]:


data.describe


# In[21]:


team = data.describe()


# In[90]:


from matplotlib import pyplot as plt
plt.plot(test)
plt.figsize = (60,60)
plt.xlabel("March Month")
plt.ylabel("Temperature")
plt.show()


# In[114]:


test_1 = data['2017-04']


# In[115]:


test_1.describe()


# In[116]:


plt.plot(test_1)
plt.figsize = (30,30)
plt.xlabel("April Month")
plt.ylabel("Temperature")
plt.show()


# In[117]:


from scipy.spatial import distance


# In[133]:


d = distance.euclidean(ravi,suren)
print("The euclidean distance between 2017/03/01 and 2017/04/01 is", d)


# In[121]:


data.head()


# In[123]:


plt.plot(testing)
plt.figsize = (100,100)
plt.xlabel("Date")
plt.ylabel("Temperature")
plt.show()


# In[134]:


from scipy.spatial import distance
dist = distance.cityblock(ravi,suren)
print("The Manhattan distance between 2017/03/01 and 2017/04/01 is ",dist)


# In[99]:


ravi = test.head()


# In[125]:


ravi.head()


# In[126]:


suren = test_1.head()


# In[127]:


suren.head()


# In[128]:


dista = distance.cityblock(ravi,suren)


# In[129]:


plt.plot(ravi)
plt.figsize = (40,40)
plt.title("Manhattan Distance")
plt.show()


# In[130]:


plt.plot(suren)
plt.figsize = (40,40)
plt.title("Manhattan Distance")
plt.show()


# In[199]:


import math
cos_sim = test_fail * test_Pass
cons_sim_1 = math.floor(math.sqrt(test_fail * test_fail))
cons_sim_2 = math.floor(math.sqrt(test_Pass * test_Pass))
consin_sim = cos_sim / (cons_sim_1) * (cons_sim_2)
print("The cosine similarity between 2017/03/01 and 2017/04/01 is ", consin_sim)


# In[200]:


import math
jacc = test_Pass
jacc_1 = test_fail + test_Pass
jacc_Ans = jacc / jacc_1
print("Jaccard similarity:", jacc_Ans)


# In[191]:


from math import *
print("Simple Matching Co-efficient")
distanc_1 = test_fail
distanc_2 = test_Pass
Simple_Mat = distanc_1 + distanc_2
Simple_Div = data
Simple_Ans = Simple_Mat / distanc_1 + distanc_2
print(Simple_Ans)


# In[190]:


test_fail = 15
test_Pass = 23


# In[ ]:




