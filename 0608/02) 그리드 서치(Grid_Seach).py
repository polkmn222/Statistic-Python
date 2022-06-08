#!/usr/bin/env python
# coding: utf-8

# ### GridSearchCV - 교차 검증과 최적 하이퍼 파라미터 튜닝을 한 번에  

# In[1]:


from sklearn.tree import DecisionTreeClassifier

get_ipython().run_line_magic('pinfo', 'DecisionTreeClassifier')


# 하이퍼파라미터란? 하이퍼 파라미터는 모델링할 때 사용자가 직접 세팅해주는 값을 뜻합니다. 

# GridSearchCV는 교차 검증을 기반으로 이 하이퍼 파라미터의 최적 값을 찾게 해준다. 즉, 데이터 세트를 cross_validation을 위한 학습/테스트 세트로 자동으로 분할한 뒤에 하이퍼 파라미터 그리드에 기술된 모든 파라미터를 순차적으로 적용해 최적의 파라미터를 찾을 수 있게 해준다.
# 
# GridSearchCV는 사용자가 튜닝하고자 하는 여러 종류의 하이퍼 파라미터를 다양하게 테스트하면서 최적의 파라미터를 편리하게 찾게 해주지만 동시에 순차적으로 파라미터를 테스트하므로 수행시간이 상대적으로 오래 걸리는 것에 유념해야 한다.

# GridSearchCV 클래스의 생성자로 들어가는 주요 파라미터는 다음과 같다.
# 
# * estimator: classifier, regressor, pipeline이 사용될 수 있다.
# 
# 
# * param_grid: key + 리스트 값을 가지는 딕셔너리가 주어진다. 
# 
# 
# * scoring: 예측 성능을 측정할 평가 방법을 지정한다. 보통은 사이킷런의 성능 평가 지표를 지정하는 문자열(예:정확도의 경우 'accuracy')로 지정하나 별도의 성능 평가 지표 함수도 지정할 수 있다.
# 
# 
# * cv:교차 검증을 위해 분할되는 학습/테스트 세트의 개수를 지정한다.
# 
# 
# * refit: 디폴트가 True이며 True로 생성 시 가장 최적의 하이퍼 파라미터를 찾은 뒤 입력된 estimator 객체를 해당 하이퍼 파라미터로 재학습시킨다.

# In[2]:


from sklearn.datasets import load_iris # DataSet
from sklearn.tree import DecisionTreeClassifier # 의사결정나무 분류기 
from sklearn.metrics import accuracy_score # 평가지표 accuracy
from sklearn.model_selection import train_test_split # 데이터 분할


# 하이퍼파라미터 튜닝과 CV를 동시에
from sklearn.model_selection import GridSearchCV


# 데이터를 로딩하고 학습 데이터와 테스트 데이터 분리
iris_data= load_iris()
X_train,X_test, y_train, y_test = train_test_split(iris_data.data, iris_data.target,
                                                  test_size=0.2, random_state=121)

# 알고리즘 객체화
dt_clf = DecisionTreeClassifier()

### 파라미터를 딕셔너리로 지정
parameters = {'max_depth':[1,2,3],
               'min_samples_split':[2,3]}


# # 1 2
#   1 3
#   2 2
#   2 3
#   3 2
#   3 3


# In[3]:


### iris_데이터에 적용

import pandas as pd

grid_dt = GridSearchCV(dt_clf, param_grid=parameters, cv=2, refit=True)

# 위의 GridSearch로 학습
grid_dt.fit(X_train,y_train)

## 결과를 보기
scores_df = pd.DataFrame(grid_dt.cv_results_)
scores_df[['params','mean_test_score','rank_test_score','split0_test_score',
          'split1_test_score']]


# In[4]:


### Q1 주요파라미터는 무엇입니까?
### Answer :: Max_depth

### Q2 현재 살펴보면 mean_test_score는 4번과 5번이 동일하다
###    가장 우수한 파라미터 조합을 찾는다면? 무엇을 골라야 할까?
### Answer :: Max_depth : 3, min_samples_split: 2

### 모수절약의 원칙


# In[5]:


print('최적 하이퍼파라미터:', grid_dt.best_estimator_)
print('최고 정확도:', grid_dt.best_score_)


# In[6]:


# 이미 refit이라는 파라미터를 GridSearchCV에서 사용하였으므로...

best_est = grid_dt.best_estimator_

pred_grid_dt = best_est.predict(X_test)
print('테스트 정확도:', accuracy_score(y_test,pred_grid_dt))


# In[7]:


### end of files

