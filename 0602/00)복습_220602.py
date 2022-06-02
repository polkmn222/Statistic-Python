#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


titan_df = pd.read_csv('./train.csv')


# In[3]:


# default는 5줄
titan_df.head(3)


# In[4]:


# 데이터의 형태(행과 열) 및 자료의 속성(type) 확인
# df.info()

titan_df.info()


# In[5]:


### 인덱스를 살펴보자.


# In[6]:


### Series == 1개의 컬럼의 Value - 1
print(type(titan_df['Pclass']))

### Series == 1개의 컬럼의 Value - 2
print(type(titan_df.Pclass))

### DataFrame == 1개이상의 2차원 컬럼
print(type(titan_df[['Pclass']]))
titan_df[['Pclass']]


# In[7]:


titan_df.head(4)


# In[8]:


### loc 및 iloc를 기반으로 한 인덱싱
# label based indexing == loc (명칭기반 인덱싱)
titan_df.loc[2,'Name'] ### Miss. Laina
titan_df.loc[0:2,'Name':'SibSp'] ### Harris~~~~~~Miss. Laina 마지막 숫자까지 가져온다.


# In[9]:


titan_df.columns[0:5]


# In[10]:


# Positional Indexing == iloc (위치기반 인덱싱)
# iloc는 기존 파이썬의 indexing과 비슷
# 마지막 숫자는 n-1을 유지

titan_df.iloc[2,1:3]


# In[11]:


# 조건에 따른 조건

cond1 = titan_df.SibSp>0
cond2 = titan_df.Parch>0

Answer1 = titan_df[cond1&cond2].reset_index(drop=True)

print('가족과 함께 배를 탑승한 비율:', np.round(Answer1.shape[0]/titan_df.shape[0], 4))


# In[12]:


# 조건식-2
titan_df.rename(columns={'Sex':'Gender'}, inplace=True)
len(titan_df[(titan_df.Gender == 'male')&(titan_df.Age>=60)])


# In[13]:


## 인덱스를 호출
## 인덱스 참고
indexes = titan_df.index ## indexes변수를 통한 index 객체화 

print('index의 범위:', indexes)
print('index의 객체의 array값:\n', indexes.values)


# In[14]:


# Q1. games.csv를 로딩하여 lol_df에 할당/선언해주세요.


# In[40]:


lol_df = pd.read_csv('./lol_archive/games.csv')
lol_df.head(3)


# In[41]:


# Q2. 9번째 컬럼명을 출력해보세요.
lol_df.columns[8]


# In[42]:


# Q3. 9번째 컬럼의 value의 data type을 출력해보세요.
lol_df.firstBaron.dtype ## 명칭기반 인덱싱


# In[43]:


lol_df.iloc[:,8].dtype


# In[44]:


# Q4. 데이터셋의 인덱스 구성은 어떻게 되어 있는지 확인해주세요.
answer_q4 = lol_df.index
print(answer_q4)


# In[45]:


# Q5. 9번째 컬럼의 2번째 값은 무엇인지 확인해주세요. (Positional Indexing) iloc

lol_df.iloc[1,8]


# ### jeju data_frame에 의한 문제풀이

# In[46]:


# Q1. jeju.csv 데이터를 로딩하여 보세요.
import pandas as pd
jeju_df = pd.read_csv('./jeju.csv')


# In[47]:


# Q2. 데이터의 마지막 3개의 행을 출력해보세요
jeju_df.tail(3)


# In[48]:


# Q3. 데이터의 수치형 변수를 출력해보세요. - select_dtypes
jeju_df.select_dtypes(exclude='object').columns


# In[49]:


# Q4. 데이터의 문자형 변수를 출력해보세요. - select_dtypes 
jeju_df.select_dtypes(include='object').columns


# In[50]:


# Q5. 데이터의 결측치를 확인해보세요. 
jeju_df.isna().sum()


# In[51]:


# Q6. 각 컬럼의 데이터수, 데이터 타입을 한 번에 확인하라. - info
jeju_df.info()


# In[52]:


# Q7. 각 수치형 변수의 분포(사분위, 평균, 표준편차, 최대, 최소)를 확인하라. - summary
jeju_df.describe()


# In[53]:


# Q8. 거주인구 컬럼의 값들을 출력하라. (보통 이러면 Series로 출력하게 됩니다) - 컬럼 지정
jeju_df.거주인구
jeju_df['거주인구']


# In[54]:


# Q9. 평균속도 컬럼의 4분위 범위(IQR)값을 구하여라.
jeju_df['평균 속도'].quantile(0.75)-jeju_df['평균 속도'].quantile(0.25)


# In[55]:


### 데이터 프레임의 컬럼 생성 및 drop
titan_df['Age_by_0'] = 0
titan_df.head(3)


# In[56]:


### drop
titan_df1 = titan_df.drop(['Age_by_0'], axis = 1) # 지워진 상태를 새로운 변수나 기존변수에 재할당


# In[57]:


### inplace 파라미터로 조절
titan_df.drop(['Age_by_0'], axis = 1, inplace= True) # 원본 데이터를 수정


# In[58]:


# end of file

