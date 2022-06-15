#!/usr/bin/env python
# coding: utf-8

# In[21]:


import pandas as pd
import numpy as np

X_train = pd.read_csv('./Part3/204_x_train.csv')
y_train = pd.read_csv('./Part3/204_y_train.csv')
X_test = pd.read_csv('./Part3/204_x_test.csv')


# In[22]:


print("X_train.shape:" , X_train.shape)
print("y_train.shape:" , y_train.shape)
print("X_tesy.shape:" , X_test.shape)

### 아래의 셀에서 X_train + X_test = X_all로 만들 예정


# In[23]:


## pd.concat 활용
X_all = pd.concat([X_train, X_test])

### 데이터 정보 확인
X_all.info()

### object인 자료의 컬럼들만 호출
print('문자형 자료의 컬럼:\n', X_all.select_dtypes(include='object').columns)


# In[24]:


# 새변수에 문자형 자료의 컬럼들 할당
ftrs = ['Warehouse_block', 'Mode_of_Shipment', 'Product_importance', 'Gender']

## Lable Encoding 진행
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

for ftr in ftrs:
    X_all[ftr] = le.fit_transform(X_all[ftr])

X_all


# In[26]:


### 불필요속성

X_all_drop = X_all.drop(['ID'], axis = 1)

### MinMaxScaler를 적용해본다.
### MinMaxScaling을 적용하는 이유는 간단한데,
### 데이터들의 정규성 가정을 우리가 확신하지 못해서 입니다.

from sklearn.preprocessing import MinMaxScaler

mm_scaler = MinMaxScaler()
result_ndarray = mm_scaler.fit_transform(X_all_drop)

print(result_ndarray)

## 전처리된 X값들과 y값을 재정의
X_train_fin = result_ndarray[:6599]
X_test_fin = result_ndarray[6599:]
y_train_fin = y_train['Reached.on.Time_Y.N']


# In[27]:


### 학습을 수행하기 위해 train_test_split을 활용한 val data를 생성
from sklearn.model_selection import train_test_split

xtrain, xval, ytrain, yval = train_test_split(X_train_fin, y_train_fin, test_size=0.2, stratify=y_train_fin, random_state=615)


# In[28]:


### 학습할 알고리즘 호출
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import  GradientBoostingClassifier

from sklearn.metrics import accuracy_score, roc_auc_score


### rf_clf를 적용
rf_clf = RandomForestClassifier(random_state=615)
rf_clf.fit(xtrain, ytrain)
pred_rf = rf_clf.predict(xval)

accuracy_rf = accuracy_score(yval, pred_rf)
roc_auc_rf = roc_auc_score(yval, pred_rf)
## 평가지표 적용
print('rf_clf의 정확도:', np.round(accuracy_rf, 4))
print('rf_clf의 roc_auc점수:', np.round(roc_auc_rf, 4))

### gb_clf를 적용
gb_clf = GradientBoostingClassifier(random_state=615)
gb_clf.fit(xtrain, ytrain)
pred_gb = gb_clf.predict(xval)

accuracy_gb = accuracy_score(yval, pred_gb)
roc_auc_gb = roc_auc_score(yval, pred_gb)
## 평가지표 적용
print('gb_clf의 정확도:', np.round(accuracy_gb, 4))
print('gb_clf의 roc_auc점수:', np.round(roc_auc_gb, 4))


# In[30]:


### lightgbm 적용

from lightgbm import LGBMClassifier

# 400개의 분류기를 생성
lbgm_wrapper = LGBMClassifier(n_estimators=400)

evals = [(xval, yval)]
lbgm_wrapper.fit(xtrain, ytrain, early_stopping_rounds=100, eval_metric='logloss', eval_set=evals, verbose=False)  # 미리 정확도가 아닌 'logloss' 지표로 알고리즘의 over-fitting 방지 및 정확도를 살펴본다.

pred_lgbm = lbgm_wrapper.predict(xval)  # xval을 넣었을때 예상되는 y값

accuracy_lgbm = accuracy_score(yval, pred_lgbm)
roc_auc_lgbm = roc_auc_score(yval, pred_lgbm)
## 평가지표 적용
print('lgbm_clf의 정확도:', np.round(accuracy_lgbm, 4))
print('lgbm_clf의 roc_auc점수:', np.round(roc_auc_lgbm, 4))


# In[34]:


## 최종결과 제출 코드 ##

final_model = gb_clf.fit(X_train_fin, y_train_fin)
y_pred = final_model.predict(X_test_fin)

### 제출파일을 생성
submit_df = pd.DataFrame({'y_pred':y_pred}).reset_index()
submit_df

submit_df.to_csv('./220615.csv')


# In[37]:


# 분석결과의 정확도가 나오지 않으므로 우리는 X변수들을
# 재확인할 필요가 있습니다.

# ID-식별자라 제외
X_train.Warehouse_block.value_counts()

X_train.Mode_of_Shipment.value_counts()


# In[39]:


# 여러분 X값들의 확인을 위해
# 다음 파일로 넘어가겠습니다.
# 여러분들이 만약에 accuracy 값을 높이고 싶으시다면
# GridSearch 혹은 다양한 피쳐 엔지니어링이 수반되어야 합니다.


# In[40]:


# end of file

