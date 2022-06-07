#!/usr/bin/env python
# coding: utf-8

# ### Stratified K fold :: 층화 K fold

# * Stratified K 폴드는 불균형한(imbalanced) 분포도를 가진 레이블(결정 클래스) 데이터 집합을 위한 K폴드 방식입니다. 불균형한 분포도를 가진 레이블 데이터 집합은 특정 레이블 값이 특이하게 많거나 매우 적어서 값의 분포가 한쪽으로 치우치는 것을 말한다.

# 가령 대출 사기 데이터를 예측한다고 가정해보자. 이 데이터 셋은 1억 건이고, 수십 개의 피처와 대출사기 여부를 뜻하는 레이블(대출사기:1, 정상대출:0)로 구성돼 있다. 그런데 대부분의 데이터는 정상 대출일 것이다.그리고 대출 사기가 약 1000건이 있다고 한다면 전체의 0.0001%의 아주 작은 확률로 대출 사기 레이블이 존재한다. 이렇게 작은 비율로 1 레이블 값이 있다면 K 폴드로 랜덤하게 학습 및 테스트 세트의 인덱스를 고르더라도 레이블 값인 0과 1의 비율을 제대로 반영하지 못하는 경우가 쉽게 발생한다.

# 즉, 레이블 값으로 1이 특정 개별 반복별 학습/테스트 데이터 세트에는 상대적으로 많이 들어 있고, 다른 반복 학습/테스트 데이터 세트에는 그렇지 못한 결과가 발생한다. 대출 사기 레이블이 1인 레코드는 비록 건수는 작지만 알고리즘이 대출 사기를 예측하기 위한 중요한 피처 값을 가지고 있기 때문에 매우 중요한 데이터 세트이다.
# 따라서 원본 데이터와 유사한 대출 사기 레이블 값의 분포를 학습/테스트 세트에도 유지하는 게 매우 중요하다.

# In[1]:


from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

import numpy as np
import pandas as pd


# In[5]:


iris = load_iris()
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df['target'] = iris.target
iris_df.head()

# 각 값의 구성확인
iris_df['target'].value_counts()

## setosa, virsicolor, virginica 각 품종이 50개씩 존재한다.


# In[8]:


iris_df.target[1]
iris_df.target.iloc[1]


# In[9]:


# KFold 수행을 해보자.
kfold = KFold(n_splits=3)
n_iter = 0
for train_index, val_index in kfold.split(iris_df):
    n_iter = n_iter + 1  # n_iter += 1
    y_train = iris_df['target'][train_index]
    y_val = iris_df['target'][val_index]
    print('## CV:{0}'.format(n_iter))
    print('y_train 데이터 분포:\n', y_train.value_counts())
    print('y_val 데이터 분포:\n', y_val.value_counts())


# In[12]:


### 위의 방법으로 데이터가 균일하지 않으므로
from sklearn.model_selection import StratifiedKFold


skf = StratifiedKFold(n_splits=3)
n_iter = 0
for train_index, val_index in skf.split(iris_df, iris_df['target']):
    n_iter = n_iter + 1  # n_iter += 1
    y_train = iris_df['target'][train_index]
    y_val = iris_df['target'][val_index]
    print('## CV:{0}'.format(n_iter))
    print('y_train 데이터 분포:\n', y_train.value_counts())
    print('y_val 데이터 분포:\n', y_val.value_counts())


# ### 교차 검증을 보다 간편하게-cross_val_score()

# 사이킷런은 교차 검증을 좀 더 편리하게 수행할 수 있게 해주는 API를 제공한다. 대표적인 것이 cross_val_score()이다. KFold로 데이터를 학습하고 예측하는 코드를 보면 먼저 (1)폴드 세트를 설정하고 (2)for 루프에서 반복으로 학습 및 테스트 데이터의 인덱스를 추출한 뒤 (3)반복적으로 학습과 예측을 수행하고 예측 성능을 반환했다.

# cross_val_score()는 이런 일련의 과정을 한꺼번에 수행해주는 API이다. 다음은 cross_val_score()API의 선언 형태이다. cross_val_score(estimator, X, y=None, scoring=None, cv=None, n_jobs=1, verbose=0, fit_params=None, pre_dispatch=‘2*n_jobs’). 이 중 estimator, X, y, scoring, cv가 주요 파라미터이다.

# In[16]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.datasets import load_iris

iris_data = load_iris()
dt_clf=DecisionTreeClassifier(random_state=156)

data= iris_data.data
label=iris_data.target

#성능 지표는 정확도(accuracy), 교차 검증 세트는 3개
scores= cross_val_score(dt_clf,data,label,scoring='accuracy', cv=3)
print('교차 검증별 정확도:', np.round(scores, 4))
print('평균 검증별 정확도:', np.round(np.mean(scores),4))


# In[13]:


# end of file

