#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sn


# # Hierarchical Clustering

# In[ ]:


crime=pd.read_csv("D:\\data science\\assignments\\ass-7 clustering\\crime_data.csv")
crime


# In[ ]:


crime2 = crime.rename({'Unnamed: 0': 'city'}, axis=1)
crime2


# In[ ]:


crime2['Assault'].hist()


# In[ ]:


crime2.info()


# In[ ]:


def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return (x)


# In[ ]:


crime3 = norm_func(crime2.iloc[:,1:])
crime3


# In[ ]:


dendrogram = sch.dendrogram(sch.linkage(crime3, method='single'))


# In[ ]:


dendrogram = sch.dendrogram(sch.linkage(crime3, method='centroid'))


# In[ ]:


dendrogram = sch.dendrogram(sch.linkage(crime3, method='average'))


# In[ ]:


crime4 = AgglomerativeClustering(n_clusters=4, affinity = 'euclidean', linkage = 'single')
crime4


# In[ ]:


y_hc = crime4.fit_predict(crime3)
Clusters=pd.DataFrame(y_hc,columns=['Clusters'])


# In[ ]:


Clusters


# # DBSCAN clustering

# In[ ]:


from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# In[ ]:


airlines = pd.read_csv("D:\\data science\\assignments\\ass-7 clustering\\EastWestAirlines.csv");
print(airlines.head())


# In[ ]:


airlines2=airlines.drop('ID#',axis=1)
airlines3=airlines2.drop('Award?',axis=1)
airlines3


# In[ ]:


array=airlines3.values
array


# In[ ]:


stscaler = StandardScaler().fit(array)
X = stscaler.transform(array)
X


# In[ ]:


dbscan = DBSCAN(eps=0.8, min_samples=6)
dbscan.fit(X)


# In[ ]:


dbscan.labels_


# In[ ]:


cl=pd.DataFrame(dbscan.labels_,columns=['cluster'])
cl


# In[32]:


pd.concat([airlines3,cl],axis=1)


# # k-mean

# In[36]:


import pandas as pd
import matplotlib.pylab as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist 
import numpy as np


# In[37]:


crimedata = pd.read_csv("D:\\data science\\assignments\\ass-7 clustering\\crime_data.csv")
crimedata


# In[38]:


X = np.random.uniform(0,1,1000)
Y = np.random.uniform(0,1,1000)
df_xy =pd.DataFrame(columns=["X","Y"])
df_xy.X = X
df_xy.Y = Y
df_xy.plot(x="X",y = "Y",kind="scatter")


# In[39]:


X = np.random.uniform(0,1,1000)
X


# In[40]:


model1 = KMeans(n_clusters=10).fit(df_xy)

df_xy.plot(x="X",y = "Y",c=model1.labels_,kind="scatter",s=10,cmap=plt.cm.coolwarm)


# In[42]:


def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return (x)

df_norm = norm_func(crimedata.iloc[:,1:])
df_norm.head(8)  


# In[45]:


model=KMeans(n_clusters=10) 
model.fit(df_norm)

model.labels_


# In[46]:


md=pd.Series(model.labels_)  
crimedata['clust']=md               
df_norm.head()

md=pd.Series(model.labels_)
crimedata['clust']=md
df_norm.head()


# In[47]:


crimedata.iloc[:,1:7].groupby(crimedata.clust).mean()


# In[48]:


crimedata.head()


# # 2nd Dataset

# In[49]:


air = pd.read_csv("D:\\data science\\assignments\\ass-7 clustering\\EastWestAirlines.csv")
air


# In[51]:


air.info()


# In[53]:


air2=air.drop('ID#',axis=1)
air2
air2.head()


# # K-mean clustering

# In[54]:


X = np.random.uniform(0,1,1000)
Y = np.random.uniform(0,1,1000)
df_xy =pd.DataFrame(columns=["X","Y"])
df_xy.X = X
df_xy.Y = Y
df_xy.plot(x="X",y = "Y",kind="scatter")


# In[55]:


X = np.random.uniform(0,1,1000)
X


# In[57]:


def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return (x)

df_norm = norm_func(air2.iloc[:,1:])


df_norm.head(5) 


# In[58]:


model=KMeans(n_clusters=5) 
model.fit(df_norm)
model.labels_


# In[60]:


md=pd.Series(model.labels_)   
air2['clust']=md               
df_norm.head()

md=pd.Series(model.labels_)
air2['clust']=md
df_norm.head()


# In[62]:


air2.iloc[:,1:7].groupby(air2.clust).mean()


# In[64]:


air2.head()


# # DBSCAN clustering

# In[65]:


from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# In[66]:


air = pd.read_csv("D:\\data science\\assignments\\ass-7 clustering\\EastWestAirlines.csv")
air


# In[69]:


airlines.head()


# In[67]:


array=airlines3.values
array


# In[70]:


stscaler = StandardScaler().fit(array)
X = stscaler.transform(array)


# In[72]:


X


# In[73]:


dbscan = DBSCAN(eps=0.8, min_samples=6)
dbscan.fit(X)


# In[74]:


dbscan.labels_


# In[75]:


cl=pd.DataFrame(dbscan.labels_,columns=['cluster'])


# In[76]:


cl


# In[77]:


pd.concat([airlines3,cl],axis=1)


# In[ ]:





# In[ ]:





# # Hierarchical Clustering

# In[43]:


import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from matplotlib import pyplot as plt
import seaborn as sn


# In[44]:


air = pd.read_csv("D:\\data science\\assignments\\ass-7 clustering\\EastWestAirlines.csv")
air


# In[35]:


air2=airlines.drop('ID#',axis=1)
air3=airlines2.drop('Award?',axis=1)
air3


# In[ ]:


def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return (x)


# In[ ]:


air4 = norm_func(air3.iloc[:,1:])


# In[31]:


dendrogram = sch.dendrogram(sch.linkage(air4, method='complete'))


# In[30]:


dendrogram = sch.dendrogram(sch.linkage(air4, method='ward'))


# In[78]:


air5 = AgglomerativeClustering(n_clusters=6, affinity = 'euclidean', linkage = 'average')
air5


# In[ ]:




