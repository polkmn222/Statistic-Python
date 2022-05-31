#!/usr/bin/env python
# coding: utf-8

# ### 딕셔너리란??
# 어떠한 데이터는 2가지의 paired된 자료로 구성되어 나타낼 수 있다. 예를 들면 '이름'='홍길동','생일'='05월 31일', '좋아하는 연예인'= '김유하(07)'
# 
# 위와 같은 대응관계를 나타내는 자료형을 연관배열(Associate Rule) 혹은 해시(Hash)라고 합니다.

# In[31]:


### 딕셔너리 생성

dict1 = {'name': '홍길동', '생일': '0531', 'phone': '01012345678'}


# In[32]:


dict1['연예인'] = 'ysp'
dict1


# ### 딕셔너리의 쌍 추가, 삭제하기

# In[7]:


a = {1 : 'a'}
a[2] = 'b'
a['name'] = 'pey'
a[3] = [1, 2, 3]
a


# ### 딕셔너리의 요소 삭제하기

# In[8]:


del a['name'] # Label based indexing

a


# ### 딕셔너리가 활용되는 예

# 김유하 = '국민가수'
# 박영식 = '강사'
# 손흥민 = '축구'
# 귀도 = '파이썬'

# In[9]:


{'김유하': '국민가수', '박영식': '강사', '손흥민':'축구', '귀도':'파이썬'}


# ### 딕셔너리의 key를 활용해서 value를 얻기

# In[10]:


a[3]


# In[11]:


### 2번째 사례
grade = {'pey' : 10, 'julliet' : 90, 'Ann' : 50}
grade['Ann']


# ### 딕셔너리 생성시 주의점

# In[13]:


### 주의점 -1
a = {1 : 'a', 1 : 'b'}
a


# 위와같이 key가 중복이 될 경우에는 나머지 key:value는 모두 무시되면서 마지막의 key:vlaue만 존재하게 된다.

# In[16]:


### 주의점 -2
## key자리에 list는 못옵니다.
{'big_data': 100, ('python', 'stat'): [98, 63]}


# In[17]:


{'big_data': 100, ('python', 'stat'): [98, 63], ['ML', 'DL']:[100, 87]}  ## unhashable type: 'list'


# ### 딕셔너리 관련 함수들

# In[18]:


a = {'name' : 'pey', 'phone' : '010-1234-5678', 'birth' : '0531'}


# In[21]:


### key로 구성된 리스트 만들기
print(a.keys())
print(type(a.keys()))  # dict_keys


# In[22]:


print(list(a.keys()))
print(type(list(a.keys())))  # list


# In[24]:


### value 리스트 만들기(values)
print(a.values())
print(list(a.values()))


# In[26]:


### key, value 쌍 얻기(items)
print(a.items())
print(list(a.items()))


# In[28]:


a['height']


# In[29]:


### get 함수의 용례
a.get('height')  # 에러를 송출하지 않는다...
a.get('height', 'nokey')  # 왼쪽의 key값이 없을때엔 우측의 값을 가져와라.


# ### DataFrame과 리스트, 딕셔너리, 넘파이 ndarray 상호변환

# In[33]:


import numpy as np
import pandas as pd

col_name1=['col1']
list1=[1,2,3]
array1= np.array(list1)
print('array1 shape:', array1.shape)
#리스트를 이용하여 DataFrame 생성.
df_list1=pd.DataFrame(list1,columns=col_name1)
print('1차원 리스트로 만든 DataFrame:\n',df_list1)
#넘파이 ndarray를 통해 DataFrame 생성
df_array1=pd.DataFrame(array1,columns=col_name1)
print('1차원 ndarray로 만든 DataFrame:\n',df_array1)


# In[35]:


#3개의 칼럼명이 필요함.
col_name2=['col1','col2','col3']

#2행 x 3열 형태의 리스트와 ndarray 생성한 뒤 이를 DataFrame으로 변환.
list2 = [[1,2,3],
         [11,12,13]]

array2 = np.array(list2)
print('array2 shape:\n',array2.shape)
df_list2=pd.DataFrame(list2,columns=col_name2)
print('2차원 리스트로 만든 DataFrame\n:',df_list2)
df_array2=pd.DataFrame(array2,columns=col_name2)
print('2차원  ndarray로 만든 DataFrame\n:',df_array2)


# In[36]:


#key는 문자열 컬럼명으로 매핑, Value는 리스트 형(또는 ndarray) 컬럼 데이터로 매핑
dict={'col1':[1,11],'col2':[2,22],'col3':[3,33]}
df_dict = pd.DataFrame(dict)
print('딕셔너리로 만든 DataFrame:\n',df_dict)


# In[37]:


#DataFrame을 ndarray로 변환
array3=df_dict.values
print('df_dict.values 타입:', type(array3), 'df_dict.values shape:', array3.shape)
print(array3)


# In[38]:


#DataFrame을 리스트로 변환
list3=df_dict.values.tolist()
print('\n df_dict.tolist()타입:',type(list3))
print(list3)
#DataFrame을 딕셔너리로 변환
dict3=df_dict.to_dict('list')
print('\n df_dict.to_dict()타입:',type(dict3))
print(dict3)


# In[39]:


# end of file

