#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
l1 = [1, 2, 3]

l2 = [[1, 2, 3],
     [11, 22, 33]]

l3 = [[[1, 2, 3]]]


# In[3]:


array1 = np.array(l1)

print('array1 type:', type(array1))
print('array1 array 형태:', array1.shape)  # 튜플의 법칙때문에~ (3,)


# In[4]:


array2 = np.array(l2)

print('array2 type:', type(array2))
print('array2 array 형태:', array2.shape)


# In[5]:


array3 = np.array(l3)

print('array3 type:', type(array3))
print('array3 array 형태:', array3.shape)


# In[7]:


### 각 array의 차원을 살펴보면,

print('array1: {0}차원, array2: {1}차원, array3: {2}차원'.format(array1.ndim, array2.ndim, array3.ndim))


# # array의 데이터 자료 확인

# In[10]:


list1 = [1, 2, 'test']
array2 = np.array(list1)
print(array2, array2.dtype)

list2 = [1, 2, 3.4]
array3 = np.array(list2)
print(array3, array3.dtype)


# # 각 자료들의 float 및 int 변환

# In[11]:


for i in range(1, 11):
    print(i)


# In[16]:


array_int = np.array([1, 2, 3])
# 위의 array_int의 자료를 float으로 변환
array_float = array_int.astype('float64')
print(array_float, array_float.dtype)

# array_float -> array_int1 데이터 변환
array_int1 = array_float.astype('int32')
print(array_int1, array_int1.dtype)

# 과연 실수데이트를 int로 변환시킬때 소수점 뒤의 자리는?
array_float1 = np.array([1.1 , 2.3, 4.4])
array_int2 = array_float1.astype('int32')
print(array_int2, array_int2.dtype)


# ## ndarray를 편리하게 생성 - arange, zeros, ones

# In[19]:


sq_array = np.arange(10)
print(sq_array)
print(sq_array.dtype, sq_array.shape)


# In[20]:


sq_array = np.arange(4, 10)
print(sq_array)
print(sq_array.dtype, sq_array.shape)


# In[23]:


print(np.zeros((3, 2), dtype='int32'))
print()
print(np.zeros((3, 2), dtype='float32'))
print()
print(np.zeros((3, 2), dtype='float64'))


# In[24]:


zero_array = np.zeros((3, 2), dtype='float64')
print(zero_array)
print(zero_array.dtype, zero_array.shape)


# In[25]:


one_array = np.ones((3, 2), dtype = 'int32')
print(one_array)
print(one_array.dtype, one_array.shape)


# In[32]:


import pandas as pd
pd.read_csv('chipotle.csv')
pd.read_csv('C:/Users/user/Desktop/Statistic/study/titanic/train.csv')  # 절대경로
pd.read_csv('./train.csv') # 상대경로
# pd.read_csv('./test/train.csv') # 상대경로


# In[ ]:


# df = pd.read_csv(DataUrl, encoding='utf-8')
# df

# 만약 df으로 변환시
# df의 자료가 한글이 포함되어 안되는 경우
# encoding = 'utf-8' 혹은 'cp949' 혹은 'euc-kr'
# 중에서 골라서 적용한 후 사용하시면
# 확실하게 데이터를 로딩시키실 수 있습니다.


# In[27]:


pd.read_csv('gender_submission.csv')


# In[33]:


pd.read_csv('jeju.csv')
# pd.read_csv('jeju.csv', encoding='cp949')


# In[29]:


pd.read_csv('test.csv')


# In[37]:


train = pd.read_csv('train.csv')
train.info()
train.head()


# In[ ]:




