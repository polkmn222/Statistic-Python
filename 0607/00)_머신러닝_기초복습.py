#!/usr/bin/env python
# coding: utf-8

# In[2]:


# 기본라이브러리 호출

import pandas as pd
import numpy as np


# In[3]:


# 분석 라이브러리 호출
from sklearn.datasets import load_iris  # 데이터
from sklearn.model_selection import train_test_split  # 데이터 분할
from sklearn.tree import DecisionTreeClassifier  # 분류알고리즘(의사결정나무)
from sklearn.metrics import  accuracy_score as acc_ac  # 평가지표(정확도)


# In[5]:


iris = load_iris()
X_ftrs = iris.data
y_label = iris.target


# In[7]:


# df 생성
iris_df = pd.DataFrame(X_ftrs, columns= iris.feature_names)
iris_df.head()


# In[9]:


# df의 label 생성

iris_df['label'] = y_label
iris_df


# In[35]:


array1 = np.arange(1, 11)  # n-1
array2 = np.arange(21, 31)  # n-1
array3 = np.arange(31, 41)  # n-1

train_test_split(array1, array2, array3 ,test_size=0.2, random_state=11)
# train_test_split(array2, test_size=0.2, random_state=11)


# In[39]:


## train과 test를 분할
# train_test_split(iris_df['label'])

# df도, ndarray도 동일하게 train_test_split 적용이 가능합니다.
X_train, X_test, y_train, y_test = train_test_split(X_ftrs, y_label, test_size=0.2, random_state=11)


# In[40]:


## 데이터 크기를 확인
print('X_train.shape: ', X_train.shape)
print('X_test.shape: ', X_test.shape)
print('y_train.shape: ', y_train.shape)
print('y_test.shape: ', y_test.shape)


# In[41]:


# 학습 및 예측을 수행 :: fit, predict를 활용

dt_clf = DecisionTreeClassifier(random_state=11)
dt_clf.fit(X_train, y_train)

pred_dt = dt_clf.predict(X_test)  # X_test를 넣어서 도출된 예측 y


# In[43]:


# 평가지표로 평가 :: (Accuracy)

result_acc = acc_ac(y_test, pred_dt)
print('의사결정 나무 정확도 :', np.round(result_acc, 4))


# In[44]:


# end of file

