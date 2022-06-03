#!/usr/bin/env python
# coding: utf-8

# ## 01 사이킷런 소개와 특징

# 사이킷런(scikit-learn)은 파이선 머신러닝 라이브러리 중 가장 많이 사용되는 라이브러리이다. 비록 최근에 텐서플로, 케라스 등 딥러닝 전문 라이브러리의 강세로 인해 대중적 관심이 줄어들고는 있지만 여전히 많은 데이터 분석가가 의존하는 대표적 파이썬 ML라이브러리이다.

# * 파이썬 기반의 다른 머신러닝 패키지도 사이킷런 스타일의 API를 지향할 정도로 쉽고 가장 파이썬스러운 API를 제공한다.
# 
# 
# * 머신러닝을 위한 매우 다양한 알고리즘과 개발을 위한 편리한 프레임워크와 API를 제공한다.
# 
# 
# * 오랜 기간 실전 환경에서 검증됐으며, 매우 많은 환경에서 사용되는 성숙한 라이브러리이다.

# pip와 Anaconda의 명령어를 통해 가능하며, 가급적이면 conda로 셋업할 것을 권장한다. conda를 이용하면 사이킷런 구동에 필요한 넘파이나 사이파이 등의 다양한 라이브러리를 동시에 설치해준다. 

# ### conda install scikit-learn

# pip를 이용하면 다음과 같이 설치할 수 있다.

# ###  pip install scikit-learn

# #### 책에서 해당하는 사이킷런의 버전은 0.19.1이다. 따라서 이 책의 모든 예제는 사이킷런 0.19.1기반이다. 다른 사이킷런 버전에서는 예제의 출력결과가 조금 다룰 수 있으므로 확인이 필요하다. 참고로 version 앞뒤의 '_ _ _'언더스코어 ('_ _ _')가 2개가 있는 것이다.

# ### 핵심단어
# 1) train 학습용 데이터
# 2) validation 검정용 데이터
# 3) test 평가용 데이터, 데이터분할
# 
# 4) underfitting(과소적합)
# 5) robust(최적학습됨)
# 6) overfitting(과대적합)

# In[1]:


pip install scikit-learn


# In[2]:


import sklearn
print(sklearn.__version__)


# ## 02 첫 번째 머신러닝 만들어 보기 ㅡ 붓꽃 품종 예측하기

# 사이킷런을 통해 첫 번째로 만들어볼 머신러닝 모델은 붓꽃 데이터 세트로 붓꽃의 품종을 분류(Classification)하는 것이다. 붓꽃 데이터 세트는 꽃잎의 길이와 너비, 꽃받침의 길이와 너비 피쳐(feature)를 기반으로 꽃의 품종을 예측하기 위한 것이다. 

#  분류(Classification)는 대표적인 지도학습(Supervised Learning)방법의 하나이다. 지도학습은 학습을 위한 다양한 피처와 분류 결정값인 레이블(Label)데이터로 모델을 학습한 뒤,
# 별도의 테스트 데이터 세트에서 미지의 레이블을 예측한다. 즉 지도학습은 명확한 정답이 주어진 데이터를 먼저 학습한 뒤 미지의 정답을 예측하는 방식이다.
# 
# 이 때 학습을 위해 주어진 데이터 세트를 학습 데이터 세트, 머신러닝 모델의 예측 성능을 평가하기 위해 별도로 주어진 데이터 세트를 테스트 데이터 세트로 지칭한다. 

# In[3]:


from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


# In[5]:


import pandas as pd
import numpy as np

#붓꽃 데이터 세트를 로딩
iris=load_iris()

#iris.data는 Iris 데이터 세트에서 피처(feature)만으로 된 데이터를 numpy로 가지고 있음.
iris_data=iris.data # x값 들

#iris.target은 붓꽃 데이터 세트에서 레이블(결정 값)데이터를 numpy로 가지고 있다.
iris_label=iris.target #y값 들
print('iris target값:', iris_label)
print('iris target명:', iris.target_names)

#붓꽃 데이터 세트를 자세히 보기 위해 DataFrame으로 변환한다.
iris_df = pd.DataFrame(iris_data,columns=iris.feature_names)
iris_df['label']=iris.target
iris_df.tail()


# 다음으로 학습용 데이터와 테스트용 데이터를 분리해보자. 학습용 데이터와 테스트용 데이터는 반드시 분리해야 한다. 학습데이터로 학습된 모델이 얼마나 뛰어난 성능을 가지는지 평가하려면 테스트 데이터 세트가 필요하기 때문이다.  
# 
# 이를 위해 사이킷런은 train_test_split()API를 제공한다. train_test_split()을 이용하면 학습 데이터와 테스트 데이터를 test_size 파라미터 입 값의 비율로 쉽게 분할이 가능하다. 예를 들어 test_size =0.2로 입력 파라미터를 설정하면 전체 데이터 중 테스트 데이터가 20%, 학습데이터가 80%로 데이터를 분할한다. 먼저 train_test_split()을 호출한 후 좀 더 자세히 입력 파라미터와 변환값을 살펴보자.

# X_train, X_test, y_train, y_test = train_test_split(iris_data, iris_label, test_size=0.2, random_state=11 )

# In[13]:


X_train, X_test, y_train, y_test = train_test_split(iris_data, iris_label, test_size=0.2, random_state=11 )


# In[14]:


X_train.shape


# In[15]:


X_test.shape


# In[16]:


y_train.shape


# In[17]:


y_test.shape


# train_test_split()의 첫 번재 파라미터인 iris_data는 피처 데이터 세트이다. 두번째 파라미터인 iris_label은 레이블(Label)데이터 세트이다. 그리고 test_size =0.2는 전체 데이터 세트 중 테스트 데이터 세트의 비율이다. 마지막으로 random_state는 호출할 때마다 같은 학습/테스트 용 데이터 세트를 생성하기 위해 주어진 난수 발생 값이다. 

# 위 예제에서 train_test_split()은 학습용 피처 데이터 세트를 X_train으로, 테스트용 피처 데이터 세트를 X_test로, 학습용 레이블 데이터 학습용 레이블 데이터 세트를 y_train으로, 테스트용 레이블 데이터 세트를 y_test로 반환한다. 

# 이제 데이터를 확보했으니 이 데이터를 기반으로 머신러닝 분류 알고리즘의 하나인 의사 결정 트리를 이용해 학습과 예측을 수행해 보자. 먼저 사이킷런의 의사결정 트리 클래스인 DecisionTreeClassifier를 객체로 생성한다. (DecisionTreeClassifier 객체 생성시 입력된 random_state=11 역시 예제 코드를 수행할 때마다 동일한 학습/예측 결과를 출력하기 위한 용도로만 사용된다.)

# In[19]:


#DecisionTreeClassifier
dt_clf=DecisionTreeClassifier(random_state=11)


# In[20]:


#학습수행
dt_clf.fit(X_train,y_train)


# 이제 DecisionTreeClassifier객체는 학습데이터를 기반으로 학습이 완료되었다. 예측은 반드시 학습 데이터가 아닌 다른 데이터를 이용해야 하며, 일반적으로 테스트 데이터 세트를 이용한다.

# In[21]:


#학습이 완료된 DecisionTreeClassifier 객체에서 테스트 데이터 세트로 예측 수행.
pred=dt_clf.predict(X_test)


# 예측 결과를 기반으로 의사 결정 트리 기반의 DecisionTreeClassifier의 예측 성능을 평가해보자. 일반적으로 ML평가에는 여러가지가 있으나, 여기서는 정확도를 측정해보자. 정확도는 예측결과가 실제 레이블 값과 얼마나 정확히 맞는지를 평가하는 지표이다. 예측한 붓꽃 품종과 실제 테스트 데이터 세트의 붓꽃 품종이 얼마나 일치하는지 확인해보자.
# 사이킷런은 정확도 측정을 위해 accuracy_score()함수를 활용한다. 
# 
# 1번째 파라미터는 실제 레이블 데이터 세트(y_test 세트), 2번째 파라미터는 예측 레이블 데이터 세트를 입력하면 된다.

# In[22]:


from sklearn.metrics import accuracy_score as acc_sc
print('예측 정확도: {0}'.format(acc_sc(y_test,pred)))


# 학습한 의사 결정 트리의 알고리즘 예측 정확도가 약 0.9333(93.33%)으로 측정되었다. 앞의 붓꽃 데이터 세트로 분류를 예측한 프로세스를 정리하면 다음과 같다.

#     1. 데이터 세트 분리: 데이터를 학습 데이터와 테스트 데이터로 분리한다.
#     
#     2. 모델 학습: 학습 데이터를 기반으로 ML 알고리즘을 적용해 모델을 학습시킨다.
#     
#     3. 예측 수행: 학습된 ML 모델을 이용해 테스트 데이터의 분류 (즉, 붓꽃 종류)를 예측한다.
#     
#     4. 평가: 이렇게 예측된 결과 값과 테스트 데이터의 실제 결과를 비교해 ML 모델 성능을 평가한다.

# In[23]:


#end of file


# In[ ]:




