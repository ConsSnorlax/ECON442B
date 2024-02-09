#!/usr/bin/env python
# coding: utf-8

# # ECON441B HW5
# # XunGONG 205452646

# # 0.) Import the Credit Card Fraud Data From CCLE

# In[1]:


import pandas as pd
# from google.colab import drive
import matplotlib.pyplot as plt
import numpy as np


# In[ ]:


# drive.mount('/content/gdrive/', force_remount = True)


# In[2]:


df = pd.read_csv("fraudTest.csv")


# In[3]:


df.head()


# In[4]:


df_select = df[["trans_date_trans_time", "category", "amt", "city_pop", "is_fraud"]]

df_select["trans_date_trans_time"] = pd.to_datetime(df_select["trans_date_trans_time"])
df_select["time_var"] = [i.second for i in df_select["trans_date_trans_time"]]

X = pd.get_dummies(df_select, ["category"]).drop(["trans_date_trans_time", "is_fraud"], axis = 1)
y = df["is_fraud"]


# # 1.) Use scikit learn preprocessing to split the data into 70/30 in out of sample

# In[5]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# In[ ]:





# In[6]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3)


# In[7]:


X_test, X_holdout, y_test, y_holdout = train_test_split(X_test, y_test, test_size = .5)


# In[8]:


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_holdout = scaler.transform(X_holdout)


# # 2.) Make three sets of training data (Oversample, Undersample and SMOTE)

# In[9]:


from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE


# In[10]:


ros = RandomOverSampler()
over_X, over_y = ros.fit_resample(X_train, y_train)

rus = RandomUnderSampler()
under_X, under_y = rus.fit_resample(X_train, y_train)

smote = SMOTE()
smote_X, smote_y = smote.fit_resample(X_train, y_train)


# In[ ]:





# # 3.) Train three logistic regression models

# In[11]:


from sklearn.linear_model import LogisticRegression


# In[12]:


over_log = LogisticRegression().fit(over_X, over_y)

under_log = LogisticRegression().fit(under_X, under_y)

smote_log = LogisticRegression().fit(smote_X, smote_y)


# In[ ]:





# # 4.) Test the three models

# In[13]:


over_log.score(X_test, y_test)


# In[14]:


under_log.score(X_test, y_test)


# In[15]:


smote_log.score(X_test, y_test)


# In[ ]:


# We see SMOTE performing with higher accuracy but is ACCURACY really the best measure?


# In[ ]:





# # 5.) Which performed best in Out of Sample metrics?

# In[ ]:


# Sensitivity here in credit fraud is more important as seen from last class


# In[16]:


from sklearn.metrics import confusion_matrix


# In[17]:


y_true = y_test


# In[18]:


y_pred = over_log.predict(X_test)
cm = confusion_matrix(y_true, y_pred)
cm


# In[19]:


print("Over Sample Sensitivity : ", cm[1,1] /( cm[1,0] + cm[1,1]))


# In[20]:


y_pred = under_log.predict(X_test)
cm = confusion_matrix(y_true, y_pred)
cm


# In[21]:


print("Under Sample Sensitivity : ", cm[1,1] /( cm[1,0] + cm[1,1]))


# In[22]:


y_pred = smote_log.predict(X_test)
cm = confusion_matrix(y_true, y_pred)
cm


# In[23]:


print("SMOTE Sample Sensitivity : ", cm[1,1] /( cm[1,0] + cm[1,1]))


# In[ ]:





# # 6.) Pick two features and plot the two classes before and after SMOTE.

# In[25]:


raw_temp = pd.concat([pd.DataFrame(X_train), pd.DataFrame(y_train)], ignore_index= False, axis =1)


# In[26]:


raw_temp.columns = list(X.columns) + ["is_fraud"]


# In[27]:


#plt.scatter(raw_temp[raw_temp["is_fraud"] == 0]["amt"], raw_temp[raw_temp["is_fraud"] == 0]["city_pop"])

plt.scatter(raw_temp[raw_temp["is_fraud"] == 1]["amt"], raw_temp[raw_temp["is_fraud"] == 1]["city_pop"])
plt.legend(["Fraud", "Not Fraud"])
plt.xlabel("Amount")
plt.ylabel("Population")

plt.show()


# In[29]:


# raw_temp = pd.concat([smote_X, smote_y], axis =1)
raw_temp = pd.concat([pd.DataFrame(raw_temp, columns = X.columns), pd.DataFrame(smote_y, columns = ["is_fraud"])], axis = 1)    
raw_temp.columns = list(X.columns) + ["is_fraud"]


# In[30]:


#plt.scatter(raw_temp[raw_temp["is_fraud"] == 0]["amt"], raw_temp[raw_temp["is_fraud"] == 0]["city_pop"])

plt.scatter(raw_temp[raw_temp["is_fraud"] == 1]["amt"], raw_temp[raw_temp["is_fraud"] == 1]["city_pop"])
plt.legend([ "Not Fraud", "Fraud"])
plt.xlabel("Amount")
plt.ylabel("Population")

plt.show()


# # 7.) We want to compare oversampling, Undersampling and SMOTE across our 3 models (Logistic Regression, Logistic Regression Lasso and Decision Trees).
# 
# # Make a dataframe that has a dual index and 9 Rows.
# # Calculate: Sensitivity, Specificity, Precision, Recall and F1 score. for out of sample data.
# # Notice any patterns across perfomance for this model. Does one totally out perform the others IE. over/under/smote or does a model perform better DT, Lasso, LR?
# # Choose what you think is the best model and why. test on Holdout

# In[31]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import pandas as pd

resampling_methods = {
    "over" : RandomOverSampler(),
    "under" : RandomUnderSampler(),
    "smote" : SMOTE()
}

model_configs = {
    "LOG" : LogisticRegression(),
    "LASSO" : LogisticRegression(penalty = "l1", C = .5, solver = "liblinear"), # C is the inverse of the regularization strength, the regularization strength is also known as lambda
    "DicisionTree" : DecisionTreeClassifier()
}


# In[40]:


def calc_perf_metrics(y_true, y_pred):
    

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp /(tp+fn)
    specificity = tn /(tn+fp)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return (sensitivity, specificity, precision, recall, f1)
#     print(f"Precision : {tp / (tp + fp)}")
#     print(f"Recall : {tp / (tp + fn)}")
#     print(f"F1 Score : {2 * (tp / (tp + fp) * tp / (tp + fn)) / (tp / (tp + fp) + tp / (tp + fn))}")

trained_models = {}
results = []


for resample_key, resampler in resampling_methods.items():
    resample_X, resample_y = resampler.fit_resample(X_train, y_train)

    for model_name, model in model_configs.items():
        combined_key = f"{resample_key}_{model_name}"
        m = model.fit(resample_X, resample_y)
        
        trained_models[combined_key] = m
        y_pred = m.predict(X_test)
        
        sensitivity, specificity, precision, recall, f1 = calc_perf_metrics(y_test, y_pred)
        
        results.append({'Model': combined_key, 
                       'Sensitivity': sensitivity,
                       'Specificity': specificity,
                       'Precision': precision,
                       'Recall': recall,
                       'f1': f1})
#         print(f"{combined_key} : {trained_models[combined_key].score(X_test, y_test)}") 
#         y_pred = trained_models[combined_key].predict(X_test)
#         y_true = y_test
#         calc_perf_metrics(y_true, y_pred)
#         print("\n\n")


# In[42]:


results_df = pd.DataFrame(results)
results_df 


# From the table, there is no clue that smote, oversampling or undersampling out perform the other method. 
# 
# However, I think the desicion trees method performs better than Log and Lasso if we use F1 Score as standard. 
# 
# I think I will choose the oversampling decisiontree method since it gives the highest F1 score.

# In[46]:


# Test on holdout
m = trained_models['over_DicisionTree']
y_pred = m.predict(X_holdout)
sensitivity, specificity, precision, recall, f1 = calc_perf_metrics(y_holdout, y_pred)  
print('sensitivity',sensitivity)
print('specificity',specificity)
print('precision',precision)
print('recall',recall)
print('f1',f1)


# In[ ]:




