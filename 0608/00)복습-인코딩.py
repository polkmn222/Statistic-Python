#!/usr/bin/env python
# coding: utf-8

# # 데이터 인코딩
#     1) 레이블 인코딩(Lable Encoding)
#     2) 원핫 인코딩(One-Hot Encoding)

# # Lable Encoding

# In[1]:


from sklearn.preprocessing import LabelEncoder


# In[2]:


ysp = ['0식', '일식', '이식', '삼식', '빵식']

# LableEncoder를 객체화
le = LabelEncoder()
le.fit(ysp)
labels = le.transform(ysp)
print('인코딩 변환값:', labels)


# In[6]:


print('인코딩 클래스:', le.classes_)


# In[7]:


print('디코딩 원본 값:', le.inverse_transform([0, 4, 3, 2, 1]))

# Lable Encoding은 선형회귀 : y = a * x + b 및 SVM과 같이 가중치 개념의 함수에는 적용할 수 없다.


# ### 원-핫 인코딩(One-Hot Encoding)

# 1) 입력값으로 2차원 데이터(2차원 ndarray)가 필요
# 2) 따라서 원-핫 인코딩 전에 레이블 인코딩 수행이 되어야 함
# 3) 결과는 해당 feature에만 1이 되고 나머지는에는 0이 됨

# In[8]:


from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import numpy as np

items=['TV','냉장고','전자레인지','컴퓨터','선풍기','선풍기','믹서','믹서']


# In[14]:


# 먼저 문자 -> 숫자 :: LabelEncoder를 수행
le = LabelEncoder()
labels = le.fit_transform(items)
print(labels)

# 위의 lables를 -> 2차원으로...
labels_2d = labels.reshape(-1, 1)

# 원핫 인코딩 수행
oh_e = OneHotEncoder()
labels_oh = oh_e.fit_transform(labels_2d)
labels_oh

print('oh_data:\n', labels_oh.toarray())


# pandas에는 원핫 인코딩을 쉽게 지원해주는 API가 있다. get_dummiest()가 있습니다.

# In[15]:


import pandas as pd

items=['TV','냉장고','전자레인지','컴퓨터','선풍기','선풍기','믹서','믹서']
test_df = pd.DataFrame(items, columns=['items'])
pd.get_dummies(test_df)


# In[16]:


# end of file

