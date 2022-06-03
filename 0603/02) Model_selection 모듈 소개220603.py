#!/usr/bin/env python
# coding: utf-8

# ## 학습/테스트 데이터 세트 분리 - train_test_split()

# 먼저 테스트 데이터세트를 활용하지 않고 학습(train)데이터 세트로만 학습 후 예측하면 어떤 것이 문제인지를 살펴보자. 

# In[1]:


from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# In[2]:


# 데이터 로딩
iris = load_iris()

# train 데이터만 로딩
X_train_all = iris.data
y_train_all = iris.target

# 알고리즘 로딩
dt_clf = DecisionTreeClassifier()

# 학습수행
dt_clf.fit(X_train_all, y_train_all)


# In[3]:


get_ipython().run_line_magic('pinfo', 'DecisionTreeClassifier')


# In[4]:


# 예측을 수행
pred = dt_clf.predict(X_train_all)

result_acc = accuracy_score(y_train_all,pred)
print('예측 정확도:{0:0.4f}'.format(result_acc))


# 정확도가 100%이다. 뭔가 이상하다.
# 
# 위의 예측 결과가 100% 정확한 이유는 이미 학습한 학습 데이터 세트를 기반으로 예측했기 때문이다. 즉, 모의고사를 이미 한 번 보고 답을 알고 있는 상태에서 모의고사 문제와 똑같은 본고사 문제가 출제됐기 때문이다. 따라서 예측을 수행하는 데이터 세트는 학습을 수행한 학습용 데이터 세트가 아닌 전용의 테스트 데이터 세트여야 한다. 사이킷런의 train_test_split()를 통해 원본 데이터 세트에서 학습 및 테스트 데이터 세트를 쉽게 분리할 수 있다.

# In[5]:


from sklearn.model_selection import train_test_split


# sklearn.model_selection 모듈에서 train_test_split를 로드해본다. train_test_split()는 첫 번째 파라미터로 피처 데이터 세트, 두 번째 파라미터로 레이블 데이터 세트를 입력받는다. 그리고 선택적으로 다음 파라미터를 입력받는다.

# In[6]:


get_ipython().run_line_magic('pinfo', 'train_test_split')


# * test_size: 전체 데이터에서 테스트 데이터 세트 크기를 얼마로 샘플링할 것인가를 결정한다. Default는 0.25, 즉 25%이다. 
# 
# 
# * train_size: 전체 데이터에서 학습용 데이터 세트 크기를 얼마로 샘플링할 것인가를 결정한다. test_size parameter를 통상적으로 사용하기 때문에 train_size는 잘 사용되지는 않는다.
# 
# 
# * random_state: random_state는 호출할 때마다 동일한 학습/테스트용 데이터 세트를 생성하기 위해 주어지는 난수 값이다. train_test_split()는 호출 시 무작위로 데이터를 분리하므로 random_state를 지정하지 않으면 수행할 때마다 다른 학습/테스트용 데이터를 생성한다. 실습용 예제이므로 수행할 때마다 동일한 데이터 세트로 분리하기 위해 random_state를 일정한 숫자 값으로 부여하자.
# 
# 
# * train_test_split()의 반환값은 튜플 형태이다. 순차적으로 학습용 데이터의 피처 데이터 세트, 테스트용 데이터의 피처 데이터 세트, 학습용 데이터의 레이블 데이터 세트, 테스트용 데이터의 레이블 데이터 세트가 반환된다.

# In[7]:


## iris 데이터 세트를 train_test_split()을 활용하여 테스트 데이터세트를 0.3
## random_state = 121로 변경해서 수행해보자.

from sklearn.datasets import load_iris # data
from sklearn.model_selection import train_test_split # data 분할
from sklearn.tree import DecisionTreeClassifier # 분류기
from sklearn.metrics import accuracy_score # 정확도 평가지표


# In[8]:


import numpy as np
import pandas as pd


# In[9]:


# 데이터 세트 만들기
iris_data = load_iris()

X_ftrs = iris_data.data # X값들 
y_target = iris_data.target # y값들


# In[10]:


## 아까는 없었던 train_test_split을 적용해보자.
X_train, X_test, y_train, y_test = train_test_split(X_ftrs, y_target,
                                                   test_size=0.3,
                                                   random_state=121)


# In[11]:


## 데이터가 제대로 분할되었는지 한 번 확인해보자.
print('X_train의 shape:', X_train.shape)
print('X_test의 shape:', X_test.shape)
print('y_train의 shape:', y_train.shape)
print('y_test의 shape:', y_test.shape)


# In[12]:


dt_clf.fit(X_train, y_train) # X_train과 y_train이 학습됨
pred = dt_clf.predict(X_test) # X_test를 넣어서 예측값을 도출


# In[13]:


print('예측 정확도:',np.round(accuracy_score(y_test,pred),4))


# #### 교차 검증
# 
# 앞서 알고리즘을 학습시키는 학습데이터와 이에 대한 예측 성능을 평가하기 위한 별도의 테스트용 데이터가 필요하다고 하였다. 하지만 이 방법 역시 과적합(overfitting)에 취약한 약점을 가질 수 있다.
# 
# 
# * 과적합이란 모델이 학습 데이터에만 과도하게 최적화되어, 실제 예측을 다른 데이터로 수행할 경우에는 예측 성능이 과도하게 떨어지는 것을 말한다. 
# 
# 
# 그런데 고정된 학습 데이터와 테스트 데이터로 평가를 하다보면 테스트 데이터에만 최적의 성능을 발휘할 수 있도록 편향되게 모델을 유도하는 경향이 생기게 된다. 이러한 문제점을 개선하기 위해 교차 검증을 이용해 더 다양한 학습과 평가를 수행한다. 

# 교차 검증을 좀 더 간략히 설명하자면 본고사를 치르기 전에 모의고사를 여러 번 보는 것이다. 즉, 본 고사가 테스트 데이터 세트에 대해 평가하는 거라면 모의고사는 교차 검증에서 많은 학습과 검증 세트에서 알고리즘 학습과 평가를 수행하는 것이다. ML은 데이터에 기반한다. 그리고 데이터는 이상치, 분포도, 다양한 속성값, 피처 중요도 등 여러 가지 ML에 영향을 미치는 요소를 가지고 있다. 특정 ML알고리즘에서 최적으로 동작할 수 있도록 데이터를 선별해 학습한다면 실제 데이터 양식과는 많은 차이가 있을 것이고 결국 성능 저하로 이어질 것이다.

#  * 교차 검증은 이러한 데이터 편중을 막기 위해서 별도의 여러 세트로 구성된 학습 데이터 세트와 검증 데이터 세트에서 학습과 평가를 수행하는 것이다.
#  
# 그리고 각 세트에서 수행한 평가 결과에 따라 하이퍼 파라마터 튜닝 등의 모델 최적화를 더욱 손쉽게 할 수 있다.

# 대부분의 ML 모델의 성능 평가는 교차 검증 기반으로 1차 평가를 한 뒤에 최종적으로 테스트 데이터 세트에 적용해 평가하는 프로세스이다. ML에 사용되는 데이터 세트를 세분화해서 학습, 검증, 테스트 데이터 세트로 나눌 수 있다. 테스트 데이터 세트 외에 별도의 검증 데이터 세트를 두어서 최종 평가 이전에 학습된 모델을 다양하게 평가하는데 사용한다.

# In[14]:


from sklearn.datasets import load_iris # data
from sklearn.model_selection import train_test_split # data 분할
from sklearn.tree import DecisionTreeClassifier # 분류기
from sklearn.metrics import accuracy_score # 정확도 평가지표


# In[15]:


import numpy as np
import pandas as pd


# In[16]:


from sklearn.model_selection import KFold


# In[17]:


iris = load_iris()
features = iris.data #x값
label = iris.target #y값
dt_clf = DecisionTreeClassifier(random_state=156)

#5개의 폴드 세트로 분리하는 KFold 객체와 폴드 세트별 정확도를 담을 리스트 객체 생성.
Kfold = KFold(n_splits=5)
print('붓꽃 데이터 세트 크기:', features.shape[0])


# In[18]:


features.shape


# In[19]:


for i,j in [(1,2),(3,4),(6,7)]:
#     print(i)
    print(j)


# In[20]:


iris_df = pd.DataFrame(features, columns=iris_data.feature_names)
iris_df[:10]


# In[21]:


train_y=label[33:149] #33 ~149
test_y = label[0:33] # 0~32
train_y


# In[22]:


x= [1,2,3]
x.append(10)
x


# In[23]:


# KFold 객체의 split()을 호출하게 되면, fold 별 학습용, 검증용 
# row index를 array로 받을 수 있다. 

features = iris.data #x값
label = iris.target #y값
Kfold = KFold(n_splits=5)


n_iter = 0
cv_accuracy = []

for train_index, val_index in Kfold.split(features):
    X_train, X_val = features[train_index], features[val_index]
    y_train, y_val = label[train_index], label[val_index]
    
    # 학습 및 예측
    dt_clf.fit(X_train, y_train)
    pred = dt_clf.predict(X_val)
    
    result_acc = np.round(accuracy_score(y_val,pred),4)
    n_iter = n_iter + 1
    
    print('#{0} 교차 검증 정확도:{1}, 학습데이터 크기:{2}, 검증데이터 크기{3}'.format(n_iter, result_acc,
                                                                X_train.shape[0], X_val.shape[0]))
    print('#{0} 검증 세트의 인덱스:{1}'.format(n_iter, val_index))
    cv_accuracy.append(result_acc)
    
#각 교차 검증의 정확도를 평균내보자.
print('\n ## 평균 검증 정확도',np.mean(cv_accuracy))


# In[24]:


# end of file

