#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
import pandas as pd


# In[2]:


get_ipython().system('pip install scikit-learn')
# conda install scikit-learn


# In[11]:


import sklearn
print(sklearn.__version__)


# In[8]:


## 필수라이브러리 호출

import numpy as np
import pandas as pd

from sklearn.datasets import load_iris # 데이터 로드
from sklearn.model_selection import train_test_split  ## X와 y분할
from sklearn.tree import DecisionTreeClassifier  # 알고리즘
from sklearn.metrics import accuracy_score  # 평가지표


# In[13]:


load_iris


# In[15]:


load_iris().data


# In[19]:


# load_iris()를 iris로 객체화

iris = load_iris()
X_ftrs = iris.data  # X값들 = 특성(features) = 설명변수
y_target = iris.target  # y값 = 목표변수 = 종속변수 = 반응변수

X_colums = iris.feature_names


# In[22]:


# data.frame으로 변경
iris_df = pd.DataFrame(X_ftrs, columns=X_colums)
iris_df.head()
iris_df['label'] = y_target
print(iris_df.head())
iris_df.tail()


# In[42]:


iris_df.info()


# In[24]:


# Train_data 및 Test_data 분할 :: Hold_out(홀드아웃)


# In[27]:


# array1 = np.arange(1, 11)
# array2 = np.arange(21, 31)
# X_train, X_test, y_train, y_test = train_test_split(array1, array2, test_size=0.3, random_state=0)


# In[33]:


X_train, X_test, y_train, y_test = train_test_split(X_ftrs, y_target, test_size=0.2, random_state=11)


# In[34]:


### 데이터의 크기 확인
print("X_train.shape : ", X_train.shape)
print("X_test.shape : ", X_test.shape)
print("y_train.shape : ", y_train.shape)
print("y_train.shape : ", y_test.shape)


# In[35]:


### 학습알고리즘 호출 및 객체화


# In[36]:


dt_clf = DecisionTreeClassifier(random_state=11)


# In[37]:


# 학습을 수행
dt_clf.fit(X_train, y_train)


# In[40]:


# 학습된 dt_clf객체에서 테스트 데이터로 예측을 수행
pred_dt = dt_clf.predict(X_test)


# In[41]:


## 모델을 평가
from sklearn.metrics import accuracy_score as acc_sc
result_acc = acc_sc(y_test, pred_dt)
print("예측 정확도 :{0:.4f}".format(result_acc))


# 학습한 의사 결정 트리의 알고리즘 예측 정확도가 약 0.9333(93.33%)으로 측정되었다. 앞의 붓꽃 데이터 세트로 분류를 예측한 프로세스를 정리하면 다음과 같다.

#     1. 데이터 세트 분리: 데이터를 학습 데이터와 테스트 데이터로 분리한다.
# 
#     2. 모델 학습: 학습 데이터를 기반으로 ML 알고리즘을 적용해 모델을 학습시킨다.
# 
#     3. 예측 수행: 학습된 ML 모델을 이용해 테스트 데이터의 분류 (즉, 붓꽃 종류)를 예측한다.
# 
#     4. 평가: 이렇게 예측된 결과 값과 테스트 데이터의 실제 결과를 비교해 ML 모델 성능을 평가한다.
