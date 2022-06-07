#!/usr/bin/env python
# coding: utf-8

# 늘 데이터사이언티스트는 overfitting(과대적합)에 대한 두려움을 가지고 있습니다. 그러므로 이를 줄이기 위해 정해진 훈련데이터(train)와 평가데이터(test)로만 평가하는 행위도 지양하고 있습니다.

# 이에 Train_Dataset을 다시금 쪼개어 미리 Test(이하 validation)을 수행하는 행위를 하는데 이를 교차검증(Cross Validation 이하 CV)라 합니다

# 여러 Cross Validation 방법이 있는데 대표적이며 가장 많이 활용되는 방법이 K-fold방법이라 합니다~

# ## K 폴드 교차검증- Kfold CV

# K 폴드 교차 검증은 가장 보편적인 교차 검증 기법이다. 먼저 K개의 데이터 폴드 세트를 만들어서 K번만큼 각 폴드 세트에 학습과 검증 평가를 반복적으로 수행하는 방법이다.

# 사이킷런에서는 K fold 교차 검증 프로세스를 구현하기 위해 KFold와 StratifiedKFold 클래스를 제공한다. 먼저 Kfold 클래스를 이용해 붓꽃 데이터 세트를 교차 검증하고 예측 정확도를 알아보자. 붓꽃 데이터 세트와 DecisionTreeClassifier를 다시 생성한다. 그리고 5개의 폴드 세트로 분리하는 KFold 객체를 생성한다.

# In[1]:


from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

import numpy as np
import pandas as pd


# In[16]:


iris = load_iris()
X_ftrs = iris.data  # X값
y_label = iris.target  # y값

# 알고리즘 객체화
dt_clf = DecisionTreeClassifier(random_state=156)

len(X_ftrs)


# In[5]:


### K-fold :: K는 5개입니다.

Kfold = KFold(n_splits=5)  # Train데이터를 5등분함
print('iris shape은:', X_ftrs.shape)
print('iris shape은:', X_ftrs.shape[0])


# In[8]:


X_ftrs[0:2]


# In[9]:


# 위와 같이 Kfold로 객체화 한 Kfold 객체를 split()을 호출하게 되면
# 폴드 별 학습용, 검증용, 데이터의 index를 array로 반환

# i가 train의 size :: 쉽게 큰 size
# j가 test의 size :: 쉽게 작은 size
for i, j in Kfold.split(X_ftrs):
    print(i)
    print(j)


# In[14]:


X_ftrs[29]


# In[22]:


n_iter = 0
cv_accuracy = []
for train_index, val_index in Kfold.split(X_ftrs):
    # 학습용, 검증용 데이터 추출
    X_train, X_val = X_ftrs[train_index], X_ftrs[val_index]
    y_train, y_val = y_label[train_index], y_label[val_index]
    # print(X_val)
    # 의사결정 나무 학습
    dt_clf.fit(X_train, y_train)
    # 의사결정나무 예측
    pred_dt = dt_clf.predict(X_val)
    n_iter = n_iter + 1
    # 반복시 마다 정확도 측정 (Accuracy)
    accuracy = np.round(accuracy_score(y_val, pred_dt), 4)
    # print로 결과출력
    print('\n # {0} CV 정확도 :{1}'.format(n_iter, accuracy))
    cv_accuracy.append(accuracy)

# 개별로 반복된 정확도를 평균내보자.

print('\n # 평균 CV정확도',np.mean(cv_accuracy))


# In[ ]:


### 참고해보세요 ^^
# 만약 전체 데이터가 설정되면
# dt_clf.fit(X_train_all, y_train_all)
# dt_clf.predicst(X_test_all) --> 결과값이 나와서
# accuracy_score(y_test_all, pred_dt_all)


# In[23]:


# end of file

