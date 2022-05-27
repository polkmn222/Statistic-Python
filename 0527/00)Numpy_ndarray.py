#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np  # alias = 별칭
import pandas as pd


# # Numpy 데이터의 특징

# In[3]:


# ndarray의 경우에는 포함된 모든
# 요소의 데이터 속성은 같습니다.

l1 = [1, 2, 'a', 'b']
array1 = np.array(l1)
print(array1, array1.dtype)

l2 = [1, 2, 3.4]
array2 = np.array(l2)
print(array2, array2.dtype)


# In[6]:


## 데이터 간의  자료형 변환
l3 = [1, 2, 3]
array_int = np.array(l3)
array_float = array_int.astype('float64')
print(array_float, array_float.dtype)

l4 = [1.1, 2.2, 3.3]
array_float1 = np.array(l4)
array_int1 = array_float1.astype('int32')
print(array_int1, array_int1.dtype)


# ### ndarray를 편리하게 생성 - arange, zeros, ones

# In[7]:


sq_array = np.arange(10)
print(sq_array)
print(sq_array.dtype, sq_array.shape)


# In[11]:


zero_array = np.zeros((3, 2))
print(zero_array)
print(zero_array.dtype, zero_array.shape)


# In[15]:


np.ones((2, 3), dtype='int32')


# In[16]:


one_array = np.ones((2, 3), dtype='int32')
print(one_array)
print(one_array.dtype, one_array.shape)


# In[22]:


### ndarray의 차원 및 형태변환
sq_array = np.arange(1, 13)
print(sq_array)
print('sq_array의 차원: ', sq_array.ndim)
print('sq_array의 차원: ', sq_array.shape)


# In[24]:


## sql_array를 3,4로 변환
sq_array2d = sq_array.reshape(3, 4)
print(sq_array2d)
print('sq_array2d의 차원: ', sq_array2d.ndim)
print('sq_array2d의 차원: ', sq_array2d.shape)


# In[26]:


## sql_array를 4,3로 변환
sq_array2d = sq_array.reshape(4, 3)
print(sq_array2d)
print('sq_array2d의 차원: ', sq_array2d.ndim)
print('sq_array2d의 차원: ', sq_array2d.shape)


# In[27]:


## sql_array를 4,3로 변환
sq_array2d = sq_array.reshape(-1, 3)
print(sq_array2d)
print('sq_array2d의 차원: ', sq_array2d.ndim)
print('sq_array2d의 차원: ', sq_array2d.shape)


# In[25]:


## 의도적 오류 2, 7
sq_array2d = sq_array.reshape(2, 7)
print(sq_array2d)
print('sq_array2d의 차원: ', sq_array2d.ndim)
print('sq_array2d의 차원: ', sq_array2d.shape)


# ### 넘파이의 ndarray의 데이터 세트 선택하기 - 인덱싱(Indexing)

# *넘파이에서 ndarray내의 일부 데이터 셋이나 특정 데이터만을 선택 할 수 있도록 하는 인덱싱에 대한 알아보자

# #### 1.특정한 데이터만 추출
# #### 2.슬라이싱(Slicing)
# #### 3.팬시 인덱싱(Fancy Indexing)
# #### 4.불린 인덱싱(Boolean Indexing)

# In[32]:


# 1 ~ 9까지의 1차원 ndarray를 생성
array1 = np.arange(start=1, stop=10)
print('array1: ',array1)
value = array1[2]
print('value:', value)  # 잊지말자 0부터
print(type(value))


# In[39]:


### 2차원의 ndarray - 특정데이터 추출

array1d = np.arange(1, 10)
array2d = array1d.reshape(3, 3)
print(array2d)

print('(row=0, col=0) index 가리키는 값:', array2d[0,0])  # 1
print('(row=0, col=1) index 가리키는 값:', array2d[0,1])  # 2
print('(row=1, col=0) index 가리키는 값:', array2d[1,0])  # 4
print('(row=2, col=2) index 가리키는 값:', array2d[2,2])  # 9


# ### 2. 슬라이싱으로 데이터 추출

# In[47]:


array1d = np.arange(1, 10)
array2d = array1d.reshape(-3, 3)

print(array2d)
print()
print('array2d:\n', array2d[0:2, 0:2])
print()
print('array2d:\n', array2d[1:3, 1:3])
print()
print('array2d:\n', array2d[1:,:])
print()
print('array2d:\n', array2d[:,:])


# In[48]:


print('array2d:\n', array2d[0:2, 0])    # 1차원
print('array2d:\n', array2d[0:2, 0:1])  # 2차원


# ### 3.팬시 인덱싱(Fancy Indexing)¶

# In[49]:


array1d = np.arange(start=1, stop=10)
array2d = array1d.reshape(3,3)

array3 = array2d[[0,1],2]
print('array2d[[0,1],2]=>',array3.tolist())

array4 = array2d[[0,1],0:2]
print('array2d[[0,1],0:2]=>',array4.tolist())

array5 = array2d[[0,1]]
print('array2d[[0,1]]=>',array5.tolist())

array2d


# In[50]:


# end of file

