#!/usr/bin/env python
# coding: utf-8

# # ECON441B HW7
# ## Xun GONG 205452646

# In[1]:


pip install openai wikipedia


# In[3]:


import openai
import os
import wikipedia


# # 1.) Set up OpenAI and the enviornment
# 

# In[57]:


# DONE


# # 2.) Use the wikipedia api to get a function that pulls in the text of a wikipedia page

# In[121]:


page_titles = ['Shakespeare']
page_titile = page_titles[0]
results = wikipedia.search(page_titile)
page = wikipedia.page(results[0])
page.content


# In[80]:


def get_wikipedia_content(page_title):
    search_results = wikipedia.search(page_title)
    page = wikipedia.page(search_results[0])
    return(page.content)


# # 3.) Build a chatgpt bot that will analyze the text given and try to locate any false info

# In[119]:


def chatgpt_error_correction(text):
    chat_completion = client.chat.completions.create(
      model="gpt-4",
    messages=[
    {"role": "system", "content": "I will be giving you an article and let me know if anything is potentially false. Go with a fine tooth comb and have low sensitivity for locating potential errors"},
    {"role": "user", "content": text}
  ]
)
    return chat_completion.choices[0].message.content


# In[ ]:





# # 4.) Make a for loop and check a few wikipedia pages and return a report of any potentially false info via wikipedia

# In[113]:


def split_text(text,chunk_size = 8192):
    chunks = len(text)//8192 +1
    return ([text[i*chunk_size:(i+1)*chunk_size] for i in range(0,chunks-1)])


# In[122]:


page_titles = ['Shakespeare','Artificial Intelligence','UCLA']

for page_title in page_titles:
    print('______________'+page_title)
    text = get_wikipedia_content(page_title)
    for i in range(len(split_text(text))):  
        print(chatgpt_error_correction(split_text(text)[i]))


# In[ ]:





# 

# In[ ]:




