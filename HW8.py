#!/usr/bin/env python
# coding: utf-8

# # 0.) Import and Clean data

# In[1]:


import pandas as pd
# from google.colab import drive
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import plot_tree
from sklearn.metrics import confusion_matrix
import seaborn as sns


# In[ ]:


#drive.mount('/content/gdrive/', force_remount = True)


# In[3]:


df = pd.read_csv('bank-additional-full (1).csv', delimiter=';')


# In[4]:


df.head()


# In[5]:


df = df.drop(["default", "pdays",	"previous",	"poutcome",	"emp.var.rate",	"cons.price.idx",	"cons.conf.idx",	"euribor3m",	"nr.employed"], axis = 1)
df = pd.get_dummies(df, columns = ["loan", "job","marital","housing","contact","day_of_week", "campaign", "month", "education"],drop_first = True)


# In[6]:


df.head()


# In[7]:


y = pd.get_dummies(df["y"], drop_first = True)
X = df.drop(["y"], axis = 1)


# In[ ]:





# In[8]:


obs = len(y)
plt.bar(["No","Yes"],[len(y[y.yes==0])/obs,len(y[y.yes==1])/obs])
plt.ylabel("Percentage of Data")
plt.show()


# In[9]:


# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler().fit(X_train)

X_scaled = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# #1.) Based on the visualization above, use your expert opinion to transform the data based on what we learned this quarter

# In[10]:


###############
###TRANSFORM###
###############

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE

# ros = RandomOverSampler()
# over_X, over_y = ros.fit_resample(X_scaled, y_train)

# X_scaled = over_X
# y_train = over_y

smote = SMOTE()
smote_X, smote_y = smote.fit_resample(X_scaled, y_train)

X_scaled = smote_X
y_train = smote_y


# # 2.) Build and visualize a decision tree of Max Depth 3. Show the confusion matrix.

# In[ ]:





# In[11]:


dtree_main = DecisionTreeClassifier(max_depth = 3)
dtree_main.fit(X_scaled, y_train)


# In[13]:


fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)
plot_tree(dtree_main, filled = True, feature_names = X.columns, class_names=["No","Yes"])


#fig.savefig('imagename.png')


# # 1b.) Confusion matrix on out of sample data. Visualize and store as variable

# In[15]:


y_pred = dtree_main.predict(X_test)
y_true = y_test
cm_raw = confusion_matrix(y_true, y_pred)


# In[16]:


class_labels = ['Negative', 'Positive']

# Plot the confusion matrix as a heatmap
sns.heatmap(cm_raw, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()


# # 3.) Use bagging on your descision tree

# In[17]:


from sklearn.ensemble import BaggingClassifier


# In[18]:


decisiontree = DecisionTreeClassifier(max_depth = 3)

bagging = BaggingClassifier(base_estimator = decisiontree,
                           n_estimators = 100,
                           max_samples = 0.5,
                           max_features = 0.5)


# In[19]:


bagging.fit(X_scaled, y_train)
y_pred = bagging.predict(X_test)


y_true = y_test
cm_raw = confusion_matrix(y_true, y_pred)


# Plot the confusion matrix as a heatmap
sns.heatmap(cm_raw, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# # 4.) Boost your tree

# In[20]:


from sklearn.ensemble import AdaBoostClassifier


# In[21]:


adaboost = AdaBoostClassifier(base_estimator = decisiontree,
                           n_estimators = 100)


# In[22]:


adaboost.fit(X_scaled, y_train)


# In[23]:


y_pred = adaboost.predict(X_test)


y_true = y_test
cm_raw = confusion_matrix(y_true, y_pred)


# Plot the confusion matrix as a heatmap
sns.heatmap(cm_raw, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()


# # 5.) Create a superlearner with at least 4 base learner models. Use a logistic reg for your metalearner. Interpret your coefficients and save your CM.

# In[ ]:


pip install mlens


# In[24]:


predictions = [adaboost.predict(X_scaled), bagging.predict(X_scaled), dtree_main.predict(X_scaled)]
X_base_learners = np.column_stack(predictions)
super_learner = LogisticRegression()
super_learner.fit(X_base_learners, y_train)


# In[28]:


super_learner.coef_
# from the coefficients, we can tell that the weight of adaboost is much higher than the other two.
# we can also tell from the CWs above, the adaboost performs better.


# In[ ]:





# In[ ]:





# In[ ]:





# # 6.)

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




