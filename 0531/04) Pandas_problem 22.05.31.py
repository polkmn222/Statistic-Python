#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd


# In[1]:


# Q1. games.csv를 로딩하여 lol_df에 할당/선언해주세요.


# In[6]:


lol_df = pd.read_csv('./games.csv')
lol_df.head(2)


# In[2]:


# Q2. 7번째 컬럼명을 출력해보세요.


# In[11]:


print(lol_df.columns)
Answer = lol_df.columns[6]
print(Answer)


# In[3]:


# Q3. 7번째 컬럼의 value의 data type을 확인해주세요(위치기반 인덱싱)


# In[15]:


# 명칭기반 인덱싱 :: Label based indexing -1
# dtype 은 property이기에 ()괄호를 사용하시지 않습니다.
lol_df['firstTower'].dtype


# In[16]:


# 위치기반인덱싱 :: Positional indexing -2 
# dtype 은 property이기에 ()괄호를 사용하시지 않습니다.
lol_df.iloc[:,6].dtype


# In[4]:


# Q4. 데이터셋의 인덱스 구성은 어떻게 되어 있는지 확인해주세요.


# In[20]:


# Numpy와 Pandas에서는 property가 간혹 등장하는데
## 암기하시지 마시고 사용하면서
## error를 보시면서 손에 익게 연습해주세요 ^^

Answer = lol_df.index
print(Answer)


# In[5]:


# Q5. 7번째 컬럼의 3번째 값은 무엇인지 확인해주세요. (Positional Indexing) iloc


# In[26]:


## 기존의 pandas indexing -1번 풀이
lol_df['firstTower'][2]


# In[27]:


## iloc 기법을 통해서 [행, 열]컨셉으로 가져오는 것이
## 더 효율적입니다.-2번 풀이

lol_df.iloc[2,6]


# ### 새로운 데이터프레임(이하 df)를 통한 문제

# In[ ]:


# Q1. jeju.csv 데이터를 로딩하여 보세요.- pd.read_csv???


# In[29]:


import pandas as pd
jeju_df = pd.read_csv('./jeju.csv')


# In[7]:


# Q2. 데이터의 마지막 3개의 행을 출력해보세요.-tail?


# In[31]:


jeju_df.tail(3)


# In[8]:


# Q3. 데이터의 수치형 변수를 출력해보세요. - select_dtypes


# In[33]:


jeju_df.info()


# In[37]:


Answer_q3 = jeju_df.select_dtypes(exclude='object').columns
print(Answer_q3)


# In[9]:


# Q4. 데이터의 문자형 변수를 출력해보세요 - select_dtypes


# In[39]:


Answer_q4 = jeju_df.select_dtypes(include='object').columns
print(Answer_q4)


# In[10]:


# Q5. 각 컬럼의 결측치 숫자를 파악해주세요. - isna, count? sum? 


# In[42]:


jeju_df.isna().sum()


# In[11]:


# Q6. 각 컬럼의 데이터수, 데이터 타입을 한 번에 확인하라. - info


# In[44]:


Answer_q6 = jeju_df.info()
print(Answer_q6)


# In[12]:


# Q7. 각 수치형 변수의 분포(사분위, 평균, 표준편차, 최대, 최소)를 확인하라. - summary


# In[46]:


Answer_q7 = jeju_df.describe()
print(Answer_q7)


# In[13]:


# Q8. 거주인구 컬럼의 값들을 출력하라. (보통 이러면 Series로 출력하게 됩니다) - 컬럼 지정


# In[48]:


Anwer_q8 = jeju_df.거주인구
print(Anwer_q8)


# In[14]:


# Q9. 평균속도 컬럼의 4분위 범위(IQR)값을 구하여라. - 제가 풀어드려야 함.


# In[50]:


## 목요일에 답변
Answer_q9 = jeju_df['평균 속도'].quantile(0.75)-jeju_df['평균 속도'].quantile(0.25)
print(Answer_q9)


# In[51]:


# end of file

