#!/usr/bin/env python
# coding: utf-8

# In[15]:


# lol archive data 활ㅇ요
# Q1. 데이터를 로드해보세요. 
# 데이터는 \t기준으로 구분되어 있습니다 :: sep parameter 사용 
import pandas as pd
pd.read_csv('./lol_archive/games.csv')  # ep 파라미터로 만약 해당


# In[3]:


# Q2 데이터의 상위 5개 행을 출력해보세요  :: head 활용


# In[16]:


lol = pd.read_csv('./lol_archive/games.csv')
print(lol.head(5))


# In[4]:


# Q3 데이터의 행과 열의 개수를 파악해보세요  :: shape 활용


# In[17]:


print(lol.shape)


# In[5]:


# Q4. 전체 컬럼을 출력해보세요 :: columns 활용


# In[18]:


# from pandas import DataFrame as df
answer_q4 = lol.columns
# df = pd.read_csv('./lol_archive/games.csv', index_col=0)
print(answer_q4)


# In[6]:


# Q5. 7번째 컬럼명을 출력해보세요 :: indexing


# In[19]:


print(lol.columns[6])
# df = pd.read_csv('./lol_archive/games.csv', index_col=7)
# print(df)


# In[ ]:


# end of file

