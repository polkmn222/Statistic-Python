#!/usr/bin/env python
# coding: utf-8

# In[26]:


# 필수 라이브러리
import numpy as np
import pandas as pd

# 시각화 라이브러리
import matplotlib.pyplot as plt
import seaborn as sns


# In[27]:


# 머신러닝 라이브러리 - sklearn
from sklearn.model_selection import train_test_split

# 필요 라이브러리 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# 평가지표 - Accuracy
from sklearn.metrics import accuracy_score


# In[28]:


# 데이터 로딩 
titan_df = pd.read_csv('./train.csv')
titan_df.head(3)


# In[29]:


### 데이터 정보
print('### Data Information ### \n')
titan_df.info()


# In[30]:


## 결측치를 확인 후 적절한 값으로 대체(imputation)
titan_df.isnull().sum()


# In[31]:


### train 결측치 대체 함수화
def imputation_na(df):
    df['Age'].fillna(np.mean(df['Age']), inplace=True)
    df['Cabin'].fillna('N', inplace=True)
    df['Embarked'].fillna('N', inplace=True)
    df['Fare'].fillna(0,inplace=True)
    return df


# In[32]:


imputation_na(titan_df).head(3)


# In[33]:


### 전처리 후의 결측값 확인
titan_df.isna().sum()


# In[34]:


### 문자들을 숫자로 변환 (인코딩)
### 종속변수(y값)가 명목형 변수(0:사망, 1:생존) == Label encoding 써도 됨

titan_df.select_dtypes(include='object').columns


# In[35]:


from sklearn.preprocessing import LabelEncoder


# In[36]:


# Label_Encoder를 for문을 통해 반복해서 적용

ftrs = ['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked']
for ftr in ftrs:
    le = LabelEncoder()
    titan_df[ftr] = le.fit_transform(titan_df[ftr])   


# In[37]:


def Label_Encode_ftrs(df):
    # Label_Encoder를 for문을 통해 반복해서 적용
    ftrs = ['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked']
    for ftr in ftrs:
        le = LabelEncoder()
        df[ftr] = le.fit_transform(df[ftr])   
    return df


# In[38]:


# Label Encoding을 수행함으로써 문자를 숫자로 변환시킴

Label_Encode_ftrs(titan_df)


# In[39]:


### 불필요한 컬럼속성 제거 
def drop_ftrs(df):
    df.drop(['PassengerId','Name','Ticket','Cabin'], axis=1, inplace=True)
    return df


# In[40]:


### 앞에서 생성한 def함수들을 다 합쳐서 만들어보자
def preprocessing_ftrs(df):
    df = imputation_na(df)
    df = Label_Encode_ftrs(df)
    df = drop_ftrs(df)
    
    return df


# In[41]:


### 지금 이 작업은 train.csv로만 진행하므로 validation입니다 ^^

## 원본 데이터를 재로딩한 후, 
# Features (즉, X값)데이터와
# Label(즉, y값)데이터를 추출

titan_df = pd.read_csv('./train.csv')
y_titan_df = titan_df['Survived']
X_titan_df = titan_df.drop(['Survived'], axis=1)


# In[42]:


### 전처리가 끝난 X_ftrs
X1_titan_df = preprocessing_ftrs(X_titan_df)


# In[43]:


### 학습을 수행하기 위한
## 데이터 분할 :: train_test_split

X_train, X_val, y_train, y_val = train_test_split(X1_titan_df, y_titan_df,
                                                 random_state=11)


# ML 알고리즘인 결정 트리, 랜덤 포레스트, 로지스틱 회귀를 이용해 타이타닉 생존자를 예측해보자.
# 이 알고리즘에 대한 상세 설명은 보강시 설명하겠다.(로지스틱 회귀는 이름은 회귀지만 매우 강력한 분류 알고리즘이다.) 아쉽지만 현재는 사이킷런 기반의 머신러닝 코드에 익숙해지는데 집중해보자. 사이킷런은 결정 트리를 위해 DecisionTreeClassifier, 랜덤 포레스트를 위해 RandomForestClassifier, 로지스틱회귀를 위해 LogisticRegression 클래스를 제공한다.

# In[44]:


# 필요 라이브러리 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

### 객체화
dt_clf = DecisionTreeClassifier(random_state=11)
rf_clf = RandomForestClassifier(random_state=11)
lr_clf = LogisticRegression(random_state=11)


# In[45]:


# train과 validation을 통해서 미리
# 학습된 알고리즘 및 가장 높은 정확도의 알고리즘 선택

## dt_clf 학습
dt_clf.fit(X_train, y_train)
pred_dt = dt_clf.predict(X_val)
accuracy_dt = accuracy_score(y_val,pred_dt)

print('dt_clf의 정확도:', np.round(accuracy_dt,4))

## rf_clf 학습
rf_clf.fit(X_train,y_train)
pred_rf = rf_clf.predict(X_val)
accuracy_rf = accuracy_score(y_val,pred_rf)

print('rf_clf의 정확도:', np.round(accuracy_rf,4))

## lr_clf 학습
lr_clf.fit(X_train,y_train)
pred_lr = lr_clf.predict(X_val)
accuracy_lr = accuracy_score(y_val,pred_lr)

print('lr_clf의 정확도:', np.round(accuracy_lr,4))


# In[46]:


# GridSearchCV를 통한 알고리즘 최적화
# 의사결정나무 알고리즘 적용

from sklearn.model_selection import GridSearchCV

parameters = {'max_depth':[2,3,5,10],
             'min_samples_split':[2,3,5],
             'min_samples_leaf':[1,5,8]}


# In[47]:


## GridSearch-1

grid_dt_clf = GridSearchCV(dt_clf, param_grid=parameters, scoring='accuracy', cv=5)
grid_dt_clf.fit(X_train,y_train) # 전체 train이 아닌 validation을 분할하고
                                 # 난 뒤의 train입니다.

print('grid_dt_clf 최적 파라미터:', grid_dt_clf.best_params_)  
print('grid_dt_clf 최고 정확도:', np.round(grid_dt_clf.best_score_,4))

# GridSearch로 학습된 최적 하이퍼파라미터로 
## test 값을 예측해본다.


# In[48]:


# 하이퍼파라미터 튜닝했다고 가정하겠습니다. GridSearchCV라는 것을 씀
# Kfold 했다라고 가정하겠습니다.


# In[49]:


X_test_all = pd.read_csv('./test.csv')
X1_test_all = preprocessing_ftrs(X_test_all)


# In[50]:


X_train_all = X1_titan_df.copy() ## 전체의 X_ftrs들을 의미합니다.
y_train_all = y_titan_df.copy() ## 전체 y_label들을 의미합니다.


# In[51]:


import warnings
warnings.filterwarnings('ignore')

# 전체의 데이터로 학습을 수행한다.

lr_clf.fit(X_train_all, y_train_all)
submit_pred = lr_clf.predict(X1_test_all)


# In[53]:


### GridSearch 전의 로지스틱회귀 알고리즘

submission_df = pd.read_csv('./gender_submission.csv')
submission_df.head()

submission_df['y_pred'] = submit_pred
submission_df.head(3)

## 실제 데이터(y_test_all)와의 정확도
accuracy_score(submission_df['Survived'],submission_df['y_pred'])


# In[ ]:


### GridSearch를 통한 의사결정나무 알고리즘
## GridSearch-2
best_dt_clf = grid_dt_clf.best_estimator_

# GridSearchCV 함수에서 refit을 True로 하셨다면,
# 다시금 fit함수를 통한 학습을 수행하실 필요는 없습니다.

pred_grid_dt = best_dt_clf.predict(X_test_all)
accuracy_grid = accuracy_score(submission_df['Survived'], pred_grid_dt)
np.round(accuracy_grid,4)
print('GridSearchCV를 통한 dt_clf의 정확도:',np.round(accuracy_grid,4))


# In[ ]:





# In[ ]:





# In[ ]:


# 답안제출이 developing 된 결과로
## 다시금 제출되는 것이 좋아보입니다 ^^

submission_df.to_csv('./220608.csv')


# In[ ]:


# end of file

