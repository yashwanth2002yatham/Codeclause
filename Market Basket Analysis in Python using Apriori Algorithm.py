#!/usr/bin/env python
# coding: utf-8

# ### Yatham Yashwanth Reddy 
# ### Data Science Intern 
# ### Code Clause
# ### Market Basket Analysis in Python using Apriori Algorithm

# # Importing libraries

# In[48]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# # Loading Dataset

# In[49]:


df = pd.read_csv("Groceries_dataset[1].csv")


# In[50]:


df.shape


# In[51]:


df.describe()


# In[52]:


df.info()


# In[53]:


df.notnull().sum()


# In[54]:


df.isna().sum()


# ## NO MISSING VALUES

# In[55]:


df.head()


# In[56]:


#setting index as Date
df.set_index('Date',inplace = True)


# In[57]:


df.head()


# In[58]:


#converting date into a particular format
df.index=pd.to_datetime(df.index)


# In[59]:


df.head()


# In[60]:


df.shape


# In[61]:


#gathering information about products
total_item = len(df)
total_days = len(np.unique(df.index.date))
total_months = len(np.unique(df.index.year))
print(total_item,total_days,total_months)


# ### Total 38765 items sold in 728 days throughout 24 months

# In[62]:


plt.figure(figsize=(15,5))
sns.barplot(x = df.itemDescription.value_counts().head(20).index, y = df.itemDescription.value_counts().head(20).values, palette = 'gnuplot')
plt.xlabel('itemDescription', size = 15)
plt.xticks(rotation=45)
plt.ylabel('Count of Items', size = 15)
plt.title('Top 20 Items purchased by customers', color = 'green', size = 20)
plt.show()


# In[63]:


df['itemDescription'].value_counts()


# In[64]:


#grouping dataset to form a list of products bought by same customer on same date
df=df.groupby(['Member_number','Date'])['itemDescription'].apply(lambda x: list(x))


# In[65]:


df.head(10)


# In[66]:


#apriori takes list as an input, hence converting dtaset to a list
transactions = df.values.tolist()
transactions[:10]


# In[67]:


get_ipython().system('pip install apyori')


# In[68]:


#applying apriori
from apyori import apriori
rules = apriori(transactions, min_support=0.00030,min_confidence = 0.05,min_lift = 2,min_length = 2)
results = list(rules)
results


# In[69]:


def inspect(results):
    lhs         = [tuple(result[2][0][0])[0] for result in results]
    rhs         = [tuple(result[2][0][1])[0] for result in results]
    supports    = [result[1] for result in results]
    confidences = [result[2][0][2] for result in results]
    lifts       = [result[2][0][3] for result in results]
    return list(zip(lhs, rhs, supports, confidences, lifts))
ordered_results = pd.DataFrame(inspect(results), columns = ['Left Hand Side', 'Right Hand Side', 'Support', 'Confidence', 'Lift'])


# In[70]:


ordered_results


# In[ ]:




