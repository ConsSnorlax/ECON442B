#!/usr/bin/env python
# coding: utf-8

# # 0.) Import and Clean data

# In[102]:


import pandas as pd
# from google.colab import drive
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# In[103]:


#drive.mount('/content/gdrive/', force_remount = True)
df = pd.read_csv("Country-data.csv", sep = ",")
df


# In[6]:


df.isnull().sum()


# In[9]:


y = df["country"]
X = df.drop(["country"], axis = 1)
X


# In[10]:


scaler = StandardScaler().fit(X)
X_scaled = scaler.transform(X)


# In[ ]:





# # 1.) Fit a kmeans Model with any Number of Clusters

# In[76]:


kmeans = KMeans(n_clusters = 5, n_init=10)
kmeans.fit(X_scaled)


# # 2.) Pick two features to visualize across

# In[24]:


X.columns


# In[78]:


import matplotlib.pyplot as plt

x1_index = 0
x2_index = 1


scatter = plt.scatter(X_scaled[:, x1_index], X_scaled[:, x2_index], c=kmeans.labels_, cmap='viridis', label='Clusters')


centers = plt.scatter(kmeans.cluster_centers_[:, x1_index], kmeans.cluster_centers_[:, x2_index], marker='o', color='black', s=100, label='Centers')

plt.xlabel(X.columns[x1_index])
plt.ylabel(X.columns[x2_index])
plt.title('Scatter Plot of Customers')

# Generate legend
plt.legend()

plt.grid()
plt.show()


# In[ ]:





# In[ ]:





# # 3.) Check a range of k-clusters and visualize to find the elbow. Test 30 different random starting places for the centroid means
# 

# In[81]:


WCSSs = []
Ks = range(1,15)
for k in Ks:
    kmeans = KMeans(n_clusters = k,n_init=30, init = 'random')
    kmeans.fit(X_scaled)
    # Sum of squared distances of samples to their closest cluster center,
    WCSSs.append(kmeans.inertia_)
    
plt.plot(Ks,WCSSs)
plt.xlabel('numbers of clusters')
plt.
plt.grid()
plt.show()


# In[83]:


WCSSs =  [KMeans(n_clusters = k,n_init=30, init = 'random').fit(X_scaled).inertia_ for k in range(1,15)]
WCSSs


# In[2]:





# # 4.) Use the above work and economic critical thinking to choose a number of clusters. Explain why you chose the number of clusters and fit a model accordingly.

# According to the elbow in the plot, I think I will choose 4 or 5 to be the number of clusters since it can been seen as the corner of elbow.

# # 6.) Do the same for a silhoutte plot

# In[63]:


from sklearn.metrics import silhouette_score


# In[85]:


silhouettes = []
Ks = range(2,14)
for k in Ks:
    kmeans = KMeans(n_clusters = k,n_init=30, init = 'random')
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    silhouette_avg = silhouette_score(X_scaled, cluster_labels)
    silhouettes.append(silhouette_avg)
    
plt.plot(Ks,silhouettes)
plt.grid()


# In[ ]:





# # 7.) Create a list of the countries that are in each cluster. Write interesting things you notice.

# In[87]:


kmeans = KMeans(n_clusters = 2,n_init=30, init = 'random').fit(X_scaled)


# In[92]:


preds = pd.DataFrame(kmeans.predict(X_scaled))


# In[93]:


output= pd.concat([preds,df],axis = 1)


# In[94]:


output


# In[97]:


print('Cluster 1:')
print(output.loc[output[0]==1, 'country'])


# In[3]:





# In[ ]:


#### Write an observation


# # 8.) Create a table of Descriptive Statistics. Rows being the Cluster number and columns being all the features. Values being the mean of the centroid. Use the nonscaled X values for interprotation

# In[101]:


output


# In[98]:


output.groupby(0).mean()


# In[99]:


output.groupby(0).std()


# # 9.) Write an observation about the descriptive statistics.

# From the data, the group 0 seems to contain more develped contries, gives relatively lower child mortality than the other group, relatively higer income, gdp and life expectation than the other group. The other factors seems not devidede so much between the two group.

# In[ ]:




