#!/usr/bin/env python
# coding: utf-8

# In[12]:


import os
os.getcwd()


# In[13]:


import pandas as pd
import numpy as np


# In[14]:


df_train = pd.read_csv('./datasets/datasets/part3/304_travel_insurance_train.csv') # X_train, y_train 함께  
df_test = pd.read_csv('./datasets/datasets/part3/304_travel_insurance_test.csv')  # X_test 


# In[15]:


## df 정보 확인
df_train.info()


# In[16]:


## X_train, y_train, X_test로 데이터를 재정의

y_train = df_train['TravelInsurance'].copy()
X_train = df_train.drop(['TravelInsurance'], axis=1)

X_test = df_test.copy()
X_test.head()


# In[17]:


### 데이터 학습 전 train과 test 데이터의 shape을 확인한다.

print('X_train의 shape:', X_train.shape)
print('X_test의 shape:', X_test.shape)
print('y_train의 shape:', y_train.shape)


# In[18]:


### 결측치 확인 전
### concat을 통해 
### train과 test를 하나의 데이터로 통합

X_all = pd.concat([X_train, X_test])
X_all.isna().sum() # 결측치가 없다


# In[19]:


X_all.select_dtypes(include='object').columns


# In[20]:


### 문자를 --> 숫자로 변환... 
### 분류분석의 경우 :: Label Encoding의 형식 더 낫습니다

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

ftrs = ['Employment Type', 'GraduateOrNot', 'FrequentFlyer',
       'EverTravelledAbroad']

for ftr in ftrs:
    X_all[ftr] = le.fit_transform(X_all[ftr])
    
### 불필요컬럼제거
X_all_drop = X_all.drop(['ID'],axis=1)

### MinMaxScaling
from sklearn.preprocessing import MinMaxScaler
mm_scaler = MinMaxScaler() #객체화
X_all_fin = mm_scaler.fit_transform(X_all_drop) # 불필요속성이 제거된 후의 정규화
X_all_fin # 결과값이 ndarray이므로 불필요컬럼을 미리 삭제하였습니다. 


# In[21]:


### 다시 X_train과 X_test로 분리
X_train_fin = X_all_fin[:1490]
X_test_fin = X_all_fin[1490:]

y_train_fin = y_train.copy()
y_train_fin.shape


print('X_train_fin의 shape:', X_train_fin.shape)
print('X_test_fin의 shape:', X_test_fin.shape)
print('y_train_fin의 shape:', y_train_fin.shape)

### 데이터 분할 train_test_split 활용
from sklearn.model_selection import train_test_split
xtrain, xval, ytrain, yval = train_test_split(X_train_fin, y_train_fin,
                                             test_size = 0.2,
                                             stratify = y_train_fin,
                                             random_state=11)


# In[ ]:


# break - 분석은 내일 이어서 해드리겠습니다 ^^

