#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd

col_name1 = ['col1']
l1 = [1, 2, 3]
array1 = np.array(l1)
print('array1 shape:', array1.shape)


# In[4]:


# 리스트 -> df 생성
df_l1 = pd.DataFrame(l1, columns=col_name1)
df_l1


# In[5]:


# ndarray -> df 생성
df_array1 = pd.DataFrame(array1, columns=col_name1)
df_array1


# In[12]:


# 3개의 컬럼명을 준비 (2,3)의 행렬을 생성할 계획
col_name2 = ['col1', 'col2', 'col3']

# 2 X 3 형태의 리스트와 ndarray를 생성한 후
# 이를 df로 변환

l2 = [[1,2,3],
      [11,22,33]]

array2 = np.array(l2)
print('array2 shape:\n', array2.shape)

# 2차원 리스트로 df 생성
df_l2 = pd.DataFrame(l2, columns=col_name2)
print('2차원 리스트로 df 생성\n',df_l2)

# 2차원 ndarray로 df 생성
df_array2 = pd.DataFrame(array2, columns=col_name2)
print('2차원 ndarray로 df 생성\n',df_l2)


# In[15]:


# key는 문자열 컬럼명으로 매핑
# value는 list 또는 ndarray의 형태로 적용

dict1 = {'a':[1,11],'b':[2,22],'c':[3,33]}
df_dict1 = pd.DataFrame(dict1)
print(df_dict1)


# ### DataFrame을 ndarray로 변환

# In[16]:


### DataFrame -> ndarray

array3 = df_dict1.values
print('df_dict1.values의 타입:', type(array3))
print('df_dict1.values의 shape:', array3.shape)
print(array3)


# In[19]:


df_dict1


# In[18]:


### DataFrame -> ndarray -> list로 변환
l3 = df_dict1.values.tolist()
print('\n df_dict.tolist()타입:', type(l3))
print(l3, '\n')

### DataFrame -> dict
df_dict1.to_dict()


# In[25]:


dict3 = df_dict1.to_dict('list')


# In[26]:


print('df_dict.to_dict()타입:', type(dict3))
print(dict3,'\n')


# In[27]:


# end of file

