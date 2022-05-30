#!/usr/bin/env python
# coding: utf-8

# 
# ### 넘파이의 ndarray의 데이터 세트 선택하기 - 인덱싱(Indexing)

# ### 1.특정한 데이터만 추출
# ### 2.슬라이싱(Slicing)
# ### 3.팬시 인덱싱(Fancy Indexing)
# ### 4.불린 인덱싱(Boolean Indexing)

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


array1d = np.arange(1, 10)
print(array1d)
array1d[array1d > 5]


# In[4]:


### for 문을 통한 인덱싱
for i in array1d:
    if i > 5:
        print(i, end=' ')
    else:
        pass


# In[6]:


### 불리언 결과값으로 인덱싱

array1d > 5
indexes = np.array([False, False, False, False, False, True, True, True, True])
result = array1d[indexes]
print('불리언 인덱스로 필터링한 결과:', result)


# In[8]:


### 일반 인덱싱으로 호출
array1d
indexes2 = np.array([5, 6, 7 ,8])
array1d[indexes2]


# ### 행렬의 정렬 -sort() 와 arsort()

# In[14]:


org_array = np.array([3, 1, 9 ,5])
print('원본 행렬:', org_array)

# np.sort()로 정렬
sort_array1 = np.sort(org_array)  # 결과값이 출력 -> 원본이 안 바뀜
print('np.sort() 호출 후 반환된 정렬 행렬:', sort_array1)
print('np.sort() 호출 후 반환된 원본 행렬:', org_array)

# ndarray.sort()로 정렬
sort_array2 = org_array.sort()  # 결과값이 출력 안되면 -> 원본이 바뀜
print('ndarray.sort() 호출 후 반환된 정렬 행렬:', sort_array2)
print('ndarray.sort() 호출 후 반환된 원본 행렬:', org_array)


# In[16]:


### 내림차순 정렬
sort_desc_array = np.sort(org_array)[::-1]
print(sort_desc_array)


# ### 데이터 핸들링 - 판다스
# 
# www.kaggle.com/c/titanic/data

# In[24]:


import pandas as pd
titan_df = pd.read_csv('./train.csv')


# In[25]:


print('df의 크기', titan_df.shape)


# In[26]:


titan_df.info()


# In[27]:


titan_df.describe()


# In[31]:


### 컬럼명 변경

## 바꾸는 방법 1 :: 바뀐 상태를 reassign (재할당)
titan_df1 = titan_df.rename(columns={'Sex' : 'Gender'})
titan_df1.head()


# In[32]:


### 컬럼명 변경

## 바꾸는 방법 2 :: inplace = True or False
## 원본데이터 수정
titan_df.rename(columns={'Sex' : 'Gender'}, inplace=True)
titan_df.head()


# In[34]:


np.array([1, 2, 3]).ndim


# In[37]:


### 성별의 구성, 등급의 구성등의 value값 살펴보기
### 1차원의 데이터 형태 :: Series
### 2차원의 data frame = Series + Series

print(titan_df['Gender'].value_counts())
# titan_df.Gender
print(titan_df.Pclass.value_counts())


# In[40]:


### reset_index() 우리가 생성했었던 데이터의 index를
### 재정렬한다.

titan_df_test = titan_df['Pclass'].value_counts().reset_index()

print(type(titan_df_test)) ### 2차원의 df
print(type(titan_df['Pclass']))  ## 1차원의 series


# In[42]:


### 파생변수 생성 = 새로운 컬럼을 생성한다

titan_df['Age_by_10'] = titan_df['Age'] * 10
titan_df['Fam_no'] = titan_df['SibSp'] + titan_df.Parch + 1
titan_df


# In[43]:


### 해당 컬럼 삭제
## 변수에 상태를 저장한다 -1

titan_df1 = titan_df.drop(['Age_by_10','Fam_no'], axis = 1)
titan_df1


# In[46]:


### 해당 행 삭제

pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 15)
print('### before axis 0 drop ###')
print(titan_df.head(3))

titan_df.drop([0,1,2], axis=0, inplace=True)

print('### after axis 0 drop ###')
print(titan_df.head(3))


# In[45]:


# end of file

