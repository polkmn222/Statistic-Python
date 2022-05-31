#!/usr/bin/env python
# coding: utf-8

# # 행렬의 정렬 -sort()와 argsort()

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


org_array = np.array([3, 1, 9, 5])
print('원본 행렬:', org_array)


# In[3]:


## np.sort()로 정렬
sort_array1 = np.sort(org_array)
print('sort_array1', sort_array1)
print('org_array', org_array)

## ndarray.sort()로 정찰
sort_array2 = org_array.sort()
print('sort_array2', sort_array2)
print('org_array', org_array)


# In[4]:


### 내림차순으로 정렬
sorted_desc = np.sort(org_array)[::-1]
print(sorted_desc)


# ### Axis 축을 기준으로 하여 정렬

# In[22]:


array2d = np.array([[8, 1], [7, 12]])
print(array2d)


# In[23]:


sort_array2_axis0 = np.sort(array2d, axis=0)
print('로우 방향으로 정렬:\n', sort_array2_axis0)


# In[24]:


sort_array2_axis1 = np.sort(array2d, axis=1)
print('컬럼 방향으로 정렬:\n', sort_array2_axis1)


# ### argsort()함수 :: 정렬된 행렬의 인덱스 반환!

# In[19]:


### 오름 차순

org_array = np.array([3, 1, 9, 5])
sort_indices = np.argsort(org_array)
print(sort_indices)
print('행렬 정렬 시 원본의 행렬의 index:', sort_indices)


# In[20]:


### 내림 차순

org_array = np.array([3, 1, 9, 5])
sort_indices = np.argsort(org_array)[::-1]
print(sort_indices)
print('행렬 정렬 시 원본의 행렬의 index:', sort_indices)


# In[26]:


A= np.array([[1,2],
             [3,4]])
transpose_mat=np.transpose(A)
print('A의 전치행렬:\n', transpose_mat)

B= np.array([[1,2],
             [3,4],
             [5,6]])

transpose_mat1=np.transpose(B)
print('B의 전치행렬:\n', transpose_mat1)


# In[27]:


# end of file

