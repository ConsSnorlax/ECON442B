#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd


# # 1.) Clean the Apple Data to get a quarterly series of EPS.

# In[6]:


y = pd.read_csv("AAPL_quarterly_financials.csv")


# In[7]:


y.index = y.name


# In[8]:


y = pd.DataFrame(y.loc["BasicEPS", :]).iloc[2:,:]


# In[9]:


y.index = pd.to_datetime(y.index)


# In[10]:


# CHECK IF NAS ARE NO DIVIDEND PERIOD
y = y.sort_index().fillna(0.)


# In[11]:


y


# # 2.) Come up with 6 search terms you think could nowcast earnings. (Different than the ones I used) Add in 3 terms that that you think will not Nowcast earnings. Pull in the gtrends data

# In[12]:


from pytrends.request import TrendReq


# In[62]:


# Create pytrends object
pytrends = TrendReq(hl='en-US', tz=360)

# Set up the keywords and the timeframe
keywords = ["M1", "Covid", "Apple Watch", "Vision Pro","MacBook","Tim Cook","Boardway", "Spider Man","Lasso"]  # Add your keywords here
start_date = '2004-01-01'
end_date = '2024-01-01'

# Create an empty DataFrame to store the results
df = pd.DataFrame()

# Iterate through keywords and fetch data
for keyword in keywords:
    pytrends.build_payload([keyword], cat=0, timeframe=f'{start_date} {end_date}', geo='', gprop='')
    interest_over_time_df = pytrends.interest_over_time()
    df[keyword] = interest_over_time_df[keyword]


# In[63]:


df = df.resample("Q").mean()
df


# In[64]:


# ALIGN DATA
temp = pd.concat([y, df],axis = 1).dropna()
y = temp[["BasicEPS"]].copy()
X = temp.iloc[:,1:].copy()
temp


# # 3.) Normalize all the X data

# In[20]:


from sklearn.preprocessing import StandardScaler


# In[65]:


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# # 4.) Run a Lasso with lambda of .5. Plot a bar chart.

# In[22]:


from sklearn.linear_model import Lasso


# In[66]:


lasso = Lasso(alpha = .5)
model = lasso.fit(X_scaled, y)
dir(model)
coefficients = model.coef_


# In[67]:


coefficients


# In[68]:


import matplotlib.pyplot as plt
plt.figure(figsize = (12,5))
plt.xlabel('Search Terms')
plt.ylabel('Coefficients')
plt.title('Lasso Coefficients Plot')
plt.bar(range(len(coefficients)), coefficients, tick_label=X.columns)
plt.axhline(0, color = "red")
plt.show()


# # 5.) Do these coefficient magnitudes make sense?

# In[69]:


lasso = Lasso(alpha = .1)
model = lasso.fit(X_scaled, y)
dir(model)
coefficients = model.coef_

plt.figure(figsize = (12,5))
plt.xlabel('Search Terms')
plt.ylabel('Coefficients')
plt.title('Lasso Coefficients Plot')
plt.bar(range(len(coefficients)), coefficients, tick_label=X.columns)
plt.axhline(0, color = "red")
plt.show()


# When I set alpha equals to 0.5, all coefficients were shrank to 0, including the related terms and unrelated ters. This situation occurs partly because that we choosed high alpha 0.5, which will have more penalty on the coefficients. It may also implied that all the terms may not be very strong predictors.
# 
# So I run another Lasso regression using alpha as 0.1, and I get 3 non-zero coefficients: M1, Apple Watch and Tim Cook, which are all from the search terms I think could nowcast earnings. M1 and Apple Watch seems stronger than others in nowcasting earnings. All the terms that I think could not nowcast (Boardway, Spider Man, Lasso) were shrank into 0. 
# 
# The coefficients do magnitudes make some sense.
