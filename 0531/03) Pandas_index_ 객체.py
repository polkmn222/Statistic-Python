#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[3]:


data = {'Name' : ['Chulmin', 'Eunkyoung', 'Jiwoong', 'Soobeom'],
        'Year' : [2011, 2016, 2012, 2017],
        'Gender' : ['Male', 'Female', 'Male', 'Male']}


# In[5]:


data_df = pd.DataFrame(data, index=['one', 'two', 'three', 'four'])
data_df


# In[6]:


# Eunkyoung 을 가져와 주세요 - 1

data_df['Name'][1]


# In[7]:


# Eunkyoung 을 가져와 주세요 - 2
# iloc :: positional indexing - 위치 기반 인덱싱

data_df.iloc[1,0]


# In[8]:


# Eunkyoung 을 가져와 주세요 - 2
# loc :: label based indexing - 명칭 기반 인덱싱

data_df.loc['two','Name']


# In[14]:


print(data_df.iloc[0, 0])           # Chulmin
print(data_df.loc['one','Name'])    # Chulmin
print(data_df.loc['four','Name'])   # Soobeom
print(data_df.iloc[0:2, [0, 1]])    # Numpy에서의 Fancy indexing과 비슷
print(data_df.loc['one':'four', ['Name', 'Gender']])
print(data_df.loc[data_df['Year'] > 2014])


# In[15]:


# end of file

