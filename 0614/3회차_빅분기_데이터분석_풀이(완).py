#!/usr/bin/env python
# coding: utf-8

# In[68]:


import os
os.getcwd()


# In[69]:


import pandas as pd
import numpy as np

X_test = pd.read_csv('.//304_x_test.csv') # X_test  
X_train = pd.read_csv('.//304_x_train.csv')  # X_train 
y_train = pd.read_csv('.//304_y_train.csv')  # y_train


# In[70]:


print('X_train의 shape:',X_train.shape)
print('y_train의 shape:',y_train.shape)
print('X_test의 shape:',X_test.shape)


# In[71]:


### 데이터의 기술통계량, 요약, 정보확인
### X_train과 X_test를 결합... :: pd.concat([X_train,X_test])

X_all = pd.concat([X_train,X_test])

## 데이터 정보확인
X_all.info()

## 데이터의 기술통계량
X_all.describe()


# In[72]:


items = ['TV','냉장고','전자레인지','컴퓨터','컴퓨터','컴퓨터']

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit_transform(items)


# In[73]:


dict1 = {'big_data':['ysp','sh','sk'],
        'python':['dy','luis','fonsi']}

test_df = pd.DataFrame(dict1)
test_df

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
test_df['python'] = le.fit_transform(test_df['python'])


# In[74]:


test_df


# In[75]:


### 데이터 전처리 - 결측값 X, 문자 -> 숫자, 불필요컬럼제거 
### 문자를 숫자로 변환
# select_dtypes 함수로 object데이터와 컬럼 추출
X_all.select_dtypes(include='object').columns 

# 문자열 컬럼들을 list로 할당 
obj_ftrs = ['Employment Type', 'GraduateOrNot', 'FrequentFlyer', 'EverTravelledAbroad']
obj_ftrs

# Label Encoding을 적용 
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder() # le로 객체화 

for ftr in obj_ftrs:
    X_all[ftr] = le.fit_transform(X_all[ftr])


# In[76]:


## X_all 즉, X_train과 X_test에 해당하는 문자 -> 숫자 
X_all.head(3)


## 불필요속성 제거
X_all_drop = X_all.drop(['ID'],axis=1)


### MinMaxScaling :: 데이터의 분포가 정규분포(즉, 가우시안 분포)가 아니므로..
from sklearn.preprocessing import MinMaxScaler
mm_scaler = MinMaxScaler() # 객체화
X_all_fin = mm_scaler.fit_transform(X_all_drop)
X_all_fin.shape


# In[77]:


### 다시금 X_train 과 X_test로 분리
X_train_fin = X_all_fin[:1490]
X_train_fin.shape
X_test_fin = X_all_fin[1490:]

print('X_train_fin의 shape:', X_train_fin.shape)
print('X_test_fin의 shape:', X_test_fin.shape)

# y_train_fin = y_train.copy()
y_train_fin = y_train.drop(['ID'], axis= 1)
print('y_train_fin의 shape:', y_train_fin.shape)


# In[78]:


data_ratio1 = y_train_fin.TravelInsurance.value_counts()
total_cnt = data_ratio1[0] + data_ratio1[1]  # 전체의 y 개수
insu_cnt = data_ratio1[1]

print('0과 1의 비율', np.round(insu_cnt / total_cnt, 4))


# In[79]:


### 데이터 분할 :: train_test_split :: X_train 및 y_train을 val로 쪼개는
from sklearn.model_selection import train_test_split
xtrain, xval, ytrain, yval = train_test_split(X_train_fin, y_train_fin, test_size=0.25, stratify=y_train_fin ,random_state=614)


# In[80]:


### train_test_split의 stratify의 파라미터를 조절하여
## 최대한 데이터의 분포를 train의 갑과 동이랗게 만들어줍니다

ytrain.value_counts()
yval.value_counts()

print(396/(721+396))
print(144/(299+144))


# In[81]:


### 알고리즘 적용 - dt_clf, rf_clf, gb_clf, ada_clf ___ 트리계열
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier


# In[82]:


from sklearn.metrics import accuracy_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')


# In[83]:


# dt_clf로 객체화
dt_clf = DecisionTreeClassifier(random_state=614)
dt_clf.fit(xtrain, ytrain) # 여기에 들어가는 train은 x와 y를 분할한 데이터입니다
dt_clf.predict(xval)
pred_dt = dt_clf.predict(xval)

print('dt_clf 정확도:', accuracy_score(yval, pred_dt))
print('dt_clf roc_auc:', roc_auc_score(yval, pred_dt))


# In[84]:


# rf_clf로 객체화
rf_clf = RandomForestClassifier(random_state=614)
rf_clf.fit(xtrain, ytrain) # 여기에 들어가는 train은 x와 y를 분할한 데이터입니다
rf_clf.predict(xval)
pred_rf = rf_clf.predict(xval)

print('rf_clf 정확도:', accuracy_score(yval, pred_rf))
print('rf_clf roc_auc:', roc_auc_score(yval, pred_rf))


# In[85]:


# gb_clf로 객체화
gb_clf = GradientBoostingClassifier(random_state=614)
gb_clf.fit(xtrain, ytrain) # 여기에 들어가는 train은 x와 y를 분할한 데이터입니다
gb_clf.predict(xval)
pred_gb = gb_clf.predict(xval)

print('gb_clf 정확도:', accuracy_score(yval, pred_gb))
print('gb_clf roc_auc:', roc_auc_score(yval, pred_gb))


# In[86]:


# ada_clf로 객체화
ada_clf = AdaBoostClassifier(random_state=614)
ada_clf.fit(xtrain, ytrain) # 여기에 들어가는 train은 x와 y를 분할한 데이터입니다
ada_clf.predict(xval)
pred_ada = gb_clf.predict(xval)

print('ada_clf 정확도:', accuracy_score(yval, pred_ada))
print('ada_clf roc_auc:', roc_auc_score(yval, pred_ada))


# In[87]:


### 최종 제출 모델
# 엄밀히 xtrain + xval = 전체_X_train
# 엄밀히 ytrain + yval = 전체_y_train
#  ""   X_test_fin

final_model = GradientBoostingClassifier().fit(X_train_fin, y_train_fin)
y_pred = final_model.predict_proba(X_test_fin)  # 예측확률을 뽑음
y_pred = y_pred[:, 1] # 그 예측 확률 중 class 1인 녀석을 다시 할당


# In[88]:


### csv 제출을 위한 작업
pd.DataFrame({'y_pred':np.round(y_pred, 4)})


# In[89]:


### csv 제출
result = pd.DataFrame({'y_pred':np.round(y_pred, 4)})
result.to_csv('./220614.csv')


# In[90]:


from sklearn.metrics import accuracy_score as acc_sc
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

def get_index(y_test, pred):
    accuracy = acc_sc(y_test, pred)
    p_score = precision_score(y_test, pred)
    r_score = recall_score(y_test, pred)
    f_score = f1_score(y_test, pred)
    roc_auc = roc_auc_score(y_test, pred)

    print('정확도 : {0:.4f}, 정밀도 : {1:.4f}, Recall:{2:.4f}, F1:{3:.4f}, roc:{4: .4f}'.format(accuracy, p_score, r_score, f_score, roc_auc))


# In[91]:


## 새로운 알고리즘인 LightGBM을 사용해본다.
from lightgbm import LGBMClassifier

# n_estimators=400그루를 설정
# 일반적인 알고리즘의 객체화
lgbm_wrapper = LGBMClassifier(n_estimators=400)

# LightGBM early_stopping_rounds 확인
evals = [(xval, yval)]

## LGBM 학습 및 예측
lgbm_wrapper.fit(xtrain, ytrain, early_stopping_rounds=100, eval_metric='logloss', eval_set=evals,
                 verbose=True)
# verbose True -> 값 보임, verbose False -> 값 안보임


# In[93]:


pred_lgbm = lgbm_wrapper.predict(xval)

get_index(yval, pred_lgbm)


# In[ ]:


# end of file

