#!/usr/bin/env python
# coding: utf-8

# In[8]:


### 리스트 유형
l1 = []  # l1_1 = list()
l2 = [1, 2, 3]
l3 = ['a', 'b', 'c']
l4 = [1, 2, 'a', 'b']
l5 = [1, 2, ['life', 'is']]


# In[9]:


print(l4)
l4[-1]


# In[14]:


print(l5)
len(l5)

# q1 Life, is에서 is를 가져와주세요...
print(l5[2][1])
l7 = l5[2]
print(l7[1])


# In[11]:


### 참고
import numpy as np
print('array는 {0}차원'.format(np.array([[1, 2,3], [3, 4]]).ndim))


# In[16]:


### 참고 - 2 :: 삼중 리스트
a = [1, 2, ['a', 'b', ['Life', 'is']]]
print(len(a))
a[2][2][0]


# # 리스트의 슬라이싱

# In[20]:


a = [1, 2, 3, 4, 5, 6, 7]
print(a[0:2])  # 슬라이싱의 마지막은 n-1 입니다.
print(a[:2])


# In[19]:


a[2:]


# In[21]:


### 참고 - 3 중첩된 리스트에서의 Slicing


# In[24]:


a = [1, 2, ['a', 'b', 'c'], 3, 4]
print(a)
print(a[1:5])
print(a[2][1:])  # 중첩리스트에서도 슬라이싱이 가능!!!


# # 리스트의 연산하기

# In[26]:


### 리스트 연산 + :: concat
a = [1, 2, 3]
b = [4, 5, 6]

result = a + b
print(result)
print(b + a)


# In[29]:


### 리스트 연산 * :: rep
print(a * 4)
print(len(a * 4))


# In[31]:


# q2 아래에서 2와 'hi'를 더해주세요 ^^
a = [1, 2, 3]
# a[1] + 'hi'  # '2hi'가 안됨...
print(str(a[1]) + 'hi') # 숫자를 문자로 변환한 후 concat


# In[32]:


a = [1, 2, 3]
b = a[:2]
print(b)


# # 리스트의 수정과 삭제

# In[36]:


a = [1, 2, 3]  # declare // assign
print(a)
print(a[1])
a[1] = 100
print(a)


# In[39]:


### 참고 -4 :: tuple 및 str
## tuple의 예
t1 = (1, 2, 3)
print(t1)
print(t1[1])
# t1[1] = 100


# In[40]:


### 참고 -5 :: tuple 및 str
## str의 예
str1 = '20220526'
print(str1)
print(str1[3])
# str1[3] = '!'


# In[41]:


### 리스트의 값을 삭제
a
del a[1]  # 위치기반 인덱싱(positional indexing)


# In[42]:


print(a)


# # 리스트 관련 함수들

# In[43]:


a = [1, 2, 3]
a


# In[44]:


a.append(10)
a.append([100, 1000])
a


# In[45]:


### extend의 용례
b = [1, 2, 3]
b.extend([200, 2000])  # extend는 일반 concat과 동일한 결과
b


# In[47]:


### 리스트의 정렬(sort)
### 순서를 정렬해준다

a = ['a', 'c', 'd', 'b']
a.sort()
print(a)

num = [1, 3, 2, 4]
num.sort()
print(num)


# In[48]:


### 리스트의 순서를 반대로
l1 = ['a', 'c', 'b']
l1.reverse()
print(l1)


# In[50]:


### q3. 데이터의 내림차순을 확인
l2 = [1, 3, 2, 4]
l2.reverse()
print(l2)

l2 = [1, 3, 2, 4]
l2.sort()
l2.reverse()
print(l2)


# In[51]:


### pop :: 리스트 요소 끄집어내기
a = [1, 2, 3]
a.pop(1) # 2가 나옴
print(a)
a.pop()  # 3이 나옴
print(a)


# # 리스트 관련함수 - 후순위

# In[52]:


### index 활용
a = [1, 2, 3]
a.index(3) # 3은 2번째 위치에 있어요~
# a.index(4) # 존재하지 않은 숫자를 치면 error가 뜹니다.


# In[53]:


### insert
l1 = [1, 2, 3]
l1.insert(2, 100)
print(l1)


# In[54]:


### 리스트 요소제거(remove)
a = [1, 2, 3]
a1 = a * 2
a1.remove(3)
print(a1)


# In[55]:


# end of file

