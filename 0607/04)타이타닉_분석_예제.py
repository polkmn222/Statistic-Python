#!/usr/bin/env python
# coding: utf-8

# 

# * Passengerid: 탑승자 데이터 일련번호
# * survived: 생존 여부, 0 = 사망, 1= 생존
# * pclass: 티켓의 선실 등급, 1=일등석, 2=이등석, 3=삼등석
# * sex: 탑승자 성별
# * name: 탑승자 이름
# * Age: 탑승자 나이
# * sibsp: 같이 탑승한 형제자매 또는 배우자 인원수
# * parch: 같이 탑승한 부모님 또는 어린이 인원수
# * ticket: 티켓 번호
# * fare: 요금
# * cabin: 선실 번호
# * embarked: 중간 정착 항구 C=Cherbourg, Q=Queenstown, S=Southampton

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


titan_df = pd.read_csv('./train.csv')
titan_df.head(3)


# In[6]:


print('\n ### 학습데이터 정보 ### \n')
print(titan_df.info())


# In[8]:


## 걸축값 확인 및 대체(imputation)  cf) 데이터를 만드는 방법 :: 데이터 증강
                                 ##                       augmentation

titan_df['Age'].fillna(titan_df['Age'].mean(), inplace=True)
titan_df['Cabin'].fillna('N', inplace=True)
titan_df['Embarked'].fillna('N', inplace=True)

print('데이터 na 개수', titan_df.isnull().sum().sum())


# ### 가설1.

# In[10]:


### 성별에 따른 생존률의 차이가 있을까?
sns.barplot(x='Sex', y='Survived', data=titan_df)


# In[13]:


### 성별에 따른 생존률 확인
titan_df.groupby(by=['Sex', 'Survived'])['Survived'].count()


# In[14]:


### 성별에 따른 생존률 계산
print('여성의 생존률:', np.round(233/(233+81), 3))
print('남성의 생존률:', np.round(109/(468+109), 3))


# ### 가설2.

# In[15]:


### 부자와 가난한 사람들 중에서는 생존률이 어떨까? + 성별


# In[18]:


sns.barplot(x='Pclass', y='Survived', hue='Sex', data=titan_df)


# ### 가설3.

# 

# In[20]:


### apply lambda
def get_cat(df):
    str1 = ''
    if df <= -1: str1 = 'unknown'
    elif df <= 5: str1 = 'baby'
    elif df <= 12: str1 = 'child'
    elif df <= 18: str1 = 'teen'
    elif df <= 25: str1 = 'student'
    elif df <= 35: str1 = 'young_adult'
    elif df <= 60: str1 = 'adult'
    else: str1 = 'elderly'

    return str1


# In[22]:


### apply lambda 적용
titan_df['Age_cat'] = titan_df['Age'].apply(lambda x : get_cat(x))

# X축을 순차적으로 표현하기 위함
group_names = ['unknown', 'baby', 'child', 'teen', 'student', 'young_adult', 'adult', 'elderly']

sns.barplot(x='Age_cat', y='Survived', hue='Sex', data=titan_df)


# In[24]:


titan_df.info()


# In[26]:


### Age_cat drop하자
titan_df = titan_df.drop('Age_cat', axis=1)


# In[27]:


### 문자데이터를 숫자로 바꾼다
titan_df.select_dtypes(include='object').columns


# In[30]:


### Label vs one_hot

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
ftrs = ['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked']

for ftr in ftrs:
    le.fit(titan_df[ftr])
    titan_df[ftr] = le.transform(titan_df[ftr])

titan_df.head()


# In[34]:


### 불필요 컬럼 제거 - id, name, Ticket, Cabin

titan_df = titan_df.drop(['PassengerId','Name','Ticket','Cabin'],axis=1)


# In[36]:


### 전처리가 완료된 titan_df 데이터

y_label = titan_df['Survived']
X_ftrs = titan_df.drop(["Survived"], axis=1)


# In[37]:


### 데이터 분할을 해보자.

from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_ftrs, y_label, test_size=0.2, random_state=11)


# In[40]:


### 데이터 shape의 확인
print('X_train의 shape:', X_train.shape)
print('X_val의 shape:', X_val.shape)
print('y_train의 shape:', y_train.shape)
print('y_val의 shape:', y_val.shape)


# In[50]:


### 알고리즘 - 분류 (의사결정나무, 랜덤포레스트, 로지스틱회귀)

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings(action='ignore')


# In[46]:


### 알고리즘 객체화

dt_clf = DecisionTreeClassifier()
rf_clf = RandomForestClassifier()
lr_clf = LogisticRegression()


# In[56]:


### 각 알고리즘별 학습/예측/평가
## dt_clf 학습
dt_clf.fit(X_train, y_train)
pred_dt = dt_clf.predict(X_val)
accuracy_dt = accuracy_score(y_val, pred_dt)
print('dt_clf의 정확도:', np.round(accuracy_dt, 4))


# In[57]:


## rl_clf 학습
rf_clf.fit(X_train, y_train)
pred_rf = rf_clf.predict(X_val)
accuracy_rf = accuracy_score(y_val, pred_rf)
print('rl_clf의 정확도:{0:.4f}'.format(accuracy_rf, 4))


# In[58]:


## lr_clf 학습
lr_clf.fit(X_train, y_train)
pred_lr = lr_clf.predict(X_val)
accuracy_lr = accuracy_score(y_val, pred_lr)
print('lr_clf의 정확도:{0:.4f}'.format(accuracy_lr, 4))


# In[63]:


### 결과값을 csv로 만들어 제출하기
result_df = pd.DataFrame(pred_lr)
result_df1 = result_df.rename(columns={0:'y_pred'})
result_df2 = result_df1.reset_index()
result_df2.to_csv('./220707.csv')


# In[64]:


df_result = X_val.copy()
df_result['y_pred'] = pred_lr
df_result.head()


# In[ ]:


# end of file

