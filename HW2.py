#!/usr/bin/env python
# coding: utf-8

# # 1.) Pull in Data and Convert ot Monthly

# In[1]:


import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


apple_data = yf.download('AAPL')
df = apple_data.resample("M").last()[["Adj Close"]]


# In[3]:


df


# # 2.) Create columns. 
#   - Current Stock Price, Difference in stock price, Whether it went up or down over the next month,  option premium

# In[17]:


# Difference in Stock Price 
df['Diff'] = df['Adj Close'].diff().shift(-1)
df.head()

# Target up or down
df['Target'] = np.sign(df['Diff'])

# Option Premuim
df['Premium'] = 0.08*df['Adj Close']
df


# # 3.) Pull in X data, normalize and build a LogReg on column 2

# In[18]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


# In[19]:


X = pd.read_csv("Xdata.csv", index_col="Date", parse_dates=["Date"])


# In[24]:


y = df.loc[:"2023-09-30","Target"].copy()
df = df.loc[:"2023-09-30",:].copy()


# In[21]:


logreg = LogisticRegression()

logreg.fit(X, y)

y_pred = logreg.predict(X)


# # 4.) Add columns, prediction and profits.

# In[27]:


df['Predctions'] = y_pred


# In[39]:


df['Profit'] = 0.

# True Positives
df.loc[(df['Predctions'] == 1) & (df['Target'] == 1),'Profit'] = df['Premium']


# False Positives
df.loc[(df['Predctions'] == 1) & (df['Target'] == -1),'Profit'] = 100* df['Diff'] + df['Premium']


# # 5.) Plot profits over time

# In[40]:


plt.plot(np.cumsum(df["Profit"]))
plt.show()


# # 5.5) Skills from MQE to help Lius ventures

# I think the ability to analyze data is critical to ventures. This semester, for example, I learned about time series, machine learning, and asset pricing, most of which are important tools for learning and predicting the price of stocks or other capital. How to transform the ever-changing market information into timely and effective investment strategies is the core ability of MQE learning.

# # 6.) Create a loop that stores total profits over time

# In[46]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_norm = scaler.fit_transform(X)
log_reg = LogisticRegression().fit(X_norm , y)

log_reg.predict_proba(X_norm)


# In[ ]:


outputs = [] 

for threshold in np.arange(0,1,0.01):
    df_temp = df.copy()
    df_temp['pred'] = np.where(log_reg.decision_function(X_norm) > threshold , 1, 0)
    TP = 


# # 7.) What is the optimal threshold and plot the total profits for this model.

# In[ ]:




