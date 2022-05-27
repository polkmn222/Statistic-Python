#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
titan_df = pd.read_csv('./train.csv')
titan_df.head()


# In[2]:


### 자료확인
titan_df.info()


# In[3]:


### 기술통계학(Describe)
titan_df.describe()  # 


# In[9]:


titan_df['Age_0'] = 0

# 파생변수 = colums는 변수(variable) 입니다 ^^
titan_df['Fam_no'] = titan_df['SibSp'] + titan_df['Parch'] + 1
titan_df.head(5)


# In[12]:


# 위에서 생성된 컬럼들을 지워보자. -1 ()
titan_df1 = titan_df.drop('Age_0', axis=1)
titan_df1 = titan_df.drop(['Age_0', 'Fam_no'], axis=1)
titan_df1.head(5)


# In[15]:


# 위에서 생성된 컬럼들을 지워보자. -2 (원본 데이터를 수정)
# inplace = True
titan_df.drop(['Age_0', 'Fam_no'], axis=1, inplace = True)
titan_df.head(5)


# In[ ]:


# end of file

