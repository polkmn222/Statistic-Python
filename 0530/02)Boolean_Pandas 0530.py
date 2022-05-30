#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[13]:


titan_df = pd.read_csv('./train.csv', engine='python')
print(titan_df)


# In[14]:


pd.read_csv('./train.csv', engine='python')


# In[4]:


### 60세 이상의 데이터만 가져와주세요 ^^
### 풀이 -1
len(titan_df[titan_df['Age'] >= 60])


# In[5]:


### 60세 이상의 데이터만 가져와 주세요 ^^
### 풀이 -2
titan_df[titan_df['Age']>=60].shape[0]


# In[7]:


### 60세 이상의 데이터 중에서 Name, Age, Survived만 가져와주세요.
### 풀이 -1
new_df = titan_df[titan_df['Age']>=60]
new_df[['Name', 'Age', 'Survived']]


# In[10]:


### 60세 이상의 데이터 중에서 Name, Age, Survived만 가져와주세요.
### 풀이 -2
titan_df[titan_df['Age']>=60][['Name','Age','Survived']]


# 여러 개의 복합 조건도 결합해 적용할 수 있다.
# 
# 1) and 조건일 때는 &
# 
# 2) or 조건일 때는 |
# 
# 3) Not 조건일 때는 ~

# ### 나이가 60세 초과이고 선실등급이 1등급이며, 성별이 여성인 승객을 추출해보자. 개별조건은 ()로 묶어서 복합연산자를 활용하여 정답을 낼 수 있다.

# In[18]:


# 정답은 2~
con1 = titan_df.Age > 60
con2 = titan_df.Pclass == 1
con3 = titan_df.Sex == 'female'
print(titan_df[con1&con2&con3])


# ### 위의 문제를 반대로 하여 총 row가 889명이 출력되도록 풀이해보세요 ^^

# In[19]:


titan_df[(titan_df['Age']>60) & (titan_df['Pclass'] == 1) & (titan_df['Sex'] == 'female')]


# In[40]:


print(titan_df.shape)
titan_df[~((titan_df['Age']>60) & (titan_df['Pclass'] == 1) & (titan_df['Sex'] == 'female'))]
# type(titan_df)
# titan_df.delete(t1)
# del titan_df[t1]
# print(titan_df.shape)
# print(t1)

