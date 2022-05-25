#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 리스트
l1 = [168, 169, 187]
l2 = ['A', 'B', 168]  # (3,)


# In[12]:


# 넘파이라는 모듈을 호출
import numpy as np
print(l2)
array1 = np.array(l2)

print('array1은 {0}차원'.format(array1.ndim))  # ndim -> 차원  (3,)
print('array1의 shape:', array1.shape)  # 튜플의 형태로 출력
print('array1 :', array1)


# In[13]:


l3 = [[1, 2, 3],
      [11, 22, 33]]
print(l3)
array2 = np.array(l3)
print(array2)
print('array2의 shape:', array2.shape)  # (2, 3)
print('array2은 {0}차원'.format(array2.ndim))


# In[17]:


l4 = [[[1, 2, 3]]]
print(l4)
array3 = np.array(l4)
print(array3)
print('array3의 shape:', array3.shape)  #  (1, 1, 3)
print('array3은 {0}차원'.format(array3.ndim))


# In[14]:


t1 = ()
t2 = (1, 2, 3)
t3 = 1, 2, 3
t4 = (1, 2, ('a', 'b'))


# In[15]:


# t5 = (,1)
t5 = (1,)


# In[18]:


# Q1. 아래의 형태는 어떻게 호출되는가?

l5 = [[1, 2, 3], [11, 22]]
print(l5)
array4 = np.array(l5)
print(array4)
print('array3의 shape:', array4.shape)  #  (2,)
print('array3은 {0}차원'.format(array4.ndim))  # 1차원


# In[ ]:




