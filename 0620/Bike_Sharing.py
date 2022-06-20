#!/usr/bin/env python
# coding: utf-8

# * datetime : hourly date + timestamp
# 
# * season : 1 = 봄, 2 = 여름, 3 = 가을, 4 = 겨울
# 
# * holiday : 1 = 토,일요일의 주말을 제외한 국경일과 같은 휴일, 0 = 휴일이 아닌 날
# 
# * workingday: 1 = 토,일요일의 주말과 휴일을 제외, 0 = 주말 및 휴일
# 
# * weather:
# 
# 
#     + 1 = 맑음, 약간 구름 낀 흐림    
#     + 2 = 안개, 안개 + 흐림    
#     + 3 = 가벼운 눈, 가벼운 비 + 천둥    
#     + 4 = 심한 눈, 심한 비+ 천둥
#     
#    
# * temp = 온도(섭씨)
# 
# * atemp = 체감온도(섭씨)
# 
# * humidity = 습도
# 
# * windspeed = 풍속  
# 
# * casual = 사전 미등록자가 대여한 횟수
# 
# * registered = 사전 등록자가 대여한 횟수
# 
# * count = 대여횟수

# In[10]:


## 기본라이브러리

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[11]:


# 회귀평가지표 - 
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error


# In[12]:


# 알고리즘 - 회귀계열
from sklearn.linear_model import LinearRegression, Ridge, Lasso


# In[13]:


# 알고리즘-tree 계열 

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from lightgbm import LGBMRegressor


# In[14]:


bike_df = pd.read_csv('./bike-sharing-demand/train.csv')
bike_df


# In[15]:


bike_df.info()


# In[16]:


# 위에서 존재하는 object를 datetime으로 변환 필요
# datetime을 쪼개어 시간 데이터로 정리
# year, month, day, time

bike_df['datetime'] = bike_df['datetime'].apply(pd.to_datetime) # object -> datetime으로 변경


# In[17]:


bike_df['year'] = bike_df.datetime.apply(lambda x: x.year)
bike_df['month'] = bike_df.datetime.apply(lambda x: x.month)
bike_df['day'] = bike_df.datetime.apply(lambda x: x.day)
bike_df['hour'] = bike_df.datetime.apply(lambda x: x.hour)
bike_df


# In[18]:


bike_new_df = bike_df.drop(['datetime','casual','registered'], axis=1)
bike_new_df ## 전처리가 어느정도 되었음


# In[19]:


bike_new_df['count'].hist()


# In[26]:


# 회귀평가 지표 :: RMSLE, RMSE
# 위에서 y데이터를 histogram으로 확인해 보니
# 오른쪽 꼬리가 긴 형태를 지닌다.

## 따라서 이를 조금 더 scaling 해주고자 log를 취하고 1을 더해줄 것이다.
# 위의 식은 np.log1p()함수를 하용하면 쉽게 구현이 가능하다
def rmsle(y_test, pred):
    log_y = np.log1p(y_test)  # actual_y :: log1p를 해주는 이유
                              # infinity 문제를 해결하기 위해서
    log_pred = np.log1p(pred) # y_pred
    squared_log_error1 = (log_y - log_pred) ** 2 # 실제값과 예측값의 차이는 y개수만큼
    mean_squared_log_error1 = np.mean(squared_log_error1)
    rmsle_result = np.sqrt(mean_squared_log_error1)

    return rmsle_result


# In[27]:


### RMSE 평가지표 함수화
def rmse(y_test, pred):
    rmse_result = np.sqrt(mean_squared_error(y_test, pred))
    return rmse_result


# In[35]:


# RMSLE와 RMSE의 지표를 합쳐서 함수화해보자.
def get_eval_index(y_test, pred):
    rmsle_eval = rmsle(y_test, pred)
    rmse_eval = rmse(y_test, pred)
    # mean_apsolute_error도 포함 _우리가 보기 쉽게
    mae_eval = mean_absolute_error(y_test, pred)
    print('RMSLE:{0:.4f}, RMSE:{1:.4f}, MAE:{2:.4f}'.format(rmsle_eval, rmse_eval, mae_eval))


# In[36]:


from sklearn.model_selection import train_test_split

# data_split by using train_test_split
# y값을 설정

y_target = bike_new_df['count']
X_ftrs = bike_new_df.drop(['count'], axis=1)

## 데이터 분할 :: train_test_split

xtrain, xval, ytrain, yval = train_test_split(X_ftrs, y_target, test_size=0.3, random_state=0)

# 단순선형회귀분석
lr_reg = LinearRegression()
lr_reg.fit(xtrain, ytrain)
pred_lr_reg = lr_reg.predict(xval)

get_eval_index(yval, pred_lr_reg)


# In[51]:


### 지표를 체감하기 위해
##  check_df라는 것을 만들어본다.

check_df = pd.DataFrame(yval.values, columns=['actual_y'])
check_df['pred_y'] = pred_lr_reg
check_df['diff'] = np.abs(check_df['pred_y'] - check_df['actual_y'])
check_df.sort_values(by='diff', ascending=False).reset_index()[:10]


# In[53]:


### y_val.hist를 통해 분포를 확인
## 이 분표가 오른쪽 꼬리가 길다라는 것을 확인

yval.hist()


# In[54]:


# log_scaling
log1p_yval = np.log1p(yval)
log1p_yval.hist()


# In[58]:


### y_target을 log1p를 통해
### 전체 log_scaling을 시도

y_target_log1p = np.log1p(y_target)  # log_scaling 후의 y_target


# y_target = bike_new_df['count']
# 위의 원본 y값이 아닌 스케일링 후의 log_y를 대입

y_target_log1p = np.log1p(y_target)
X_ftrs = bike_new_df.drop(['count'], axis=1)

## 데이터 분할 :: train_test_split

xtrain, xval, ytrain_log, yval_log = train_test_split(X_ftrs, y_target_log1p, test_size=0.3, random_state=0)

# 단순선형회귀분석
lr_reg = LinearRegression()
lr_reg.fit(xtrain, ytrain_log)
pred_lr1_reg = lr_reg.predict(xval)  # 로그화가 진행된 y_pred가

# 위에서 log를 통해 스케일링을 한
# yval_log를 원상복귀 시킴
yval_exp = np.expm1(yval_log)

## 그 이후 원상태의 y_pred :: pred_lr1_reg VS yval_log를 원상복귀
pred_lr1_exp = np.expm1(pred_lr_reg)

get_eval_index(yval_exp, pred_lr1_exp)


# In[59]:


# RMSLE는 줄어들었으나
# RMSE는 오히려 늘어났다. 즉, 오차값이 늘었다는 것이다
# 이에 이 이유를 규명해보고자 시각화를 살펴보자


# In[65]:


coef = pd.Series(lr_reg.coef_, index= X_ftrs.columns)
coef_sorted = coef.sort_values(ascending=False)
sns.barplot(x=coef_sorted.values, y=coef_sorted.index)


# In[67]:


# year는 연도를 의미 하므로 범주형 자료의 느낌
# 그런데 지금 해당데이터에는 데이터 크기에 따른
# 가중치(weight = coef = 계수)가 크게 부여가 됨

bike_new_df['year'].value_counts()


# In[68]:


['year', 'month', 'day', 'hour', 'holiday', 'workingday', 'season', 'weather']


# In[71]:


# 위의 데이터들 중 범주형 자료의 경우
# 원핫 인코딩을 통해 feature들을 전처리 해줌

X_ftrs_oh = pd.get_dummies(X_ftrs, columns=['year', 'month', 'day', 'hour', 'holiday', 'workingday', 'season', 'weather'])


# In[75]:


## 원핫 인코딩이 된 X_ftrs
X_ftrs_oh

### y_target을 log1p를 통해
### 전체 log_scaling을 시도
y_target_log1p = np.log1p(y_target)  # log_scaling 후의 y_target

xtrain2, xval2, ytrain2, yval2 = train_test_split(X_ftrs_oh, y_target_log1p, test_size=0.3, random_state=0)
# Ridge, Lasso, LinearRegression 적용
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

lr_reg = LinearRegression()
ridge = Ridge(alpha=10) # L2 제곱해서 더한다
lasso = Lasso(alpha=0.01) # L2 절대값으로 더한다

models = [lr_reg, ridge, lasso]

for model in models:
    model.fit(xtrain2, ytrain2)
    pred = model.predict(xval2)
    yval_exp1 = np.expm1(yval2)
    pred_exp1 = np.expm1(pred)
    print('', model.__class__.__name__, '')
    get_eval_index(yval_exp1, pred_exp1)


# In[80]:


# one_hot_encoding 이후를 시각화를 진행
# ridge :: L2규제
coef = pd.Series(ridge.coef_, index=X_ftrs_oh.columns)
coef_sorted_r = coef.sort_values(ascending=False)[:11]
coef_sorted_r

sns.barplot(x=coef_sorted_r.values, y=coef_sorted_r.index)


# In[81]:


# one_hot_encoding 이후를 시각화를 진행
# Lasso :: L1규제
coef_l = pd.Series(lasso.coef_, index=X_ftrs_oh.columns)
coef_sorted_l = coef.sort_values(ascending=False)[:11]
coef_sorted_l

sns.barplot(x=coef_sorted_l.values, y=coef_sorted_l.index)


# In[83]:


### RandomForestRegressor 적용
from sklearn.ensemble import RandomForestRegressor

# 모델로 평가지표 분석
rf_reg = RandomForestRegressor(n_estimators=500)

rf_reg.fit(xtrain2, ytrain2)
pred_rf = rf_reg.predict(xval2)

# y 원상 복귀
yval_expm1_rf = np.expm1(yval2)

# pred 원상복귀
pred_exam1_rf = np.expm1(pred_rf)

get_eval_index(yval_expm1_rf, pred_exam1_rf)


# In[ ]:


### LGBMRegressor를 활용하여
### 위의 지표 확인 :: 수치를 적어본다.
### 아마 랜포보다 더 좋은 수치를 기대


# In[ ]:


### test 데이터를 통해 다시금
### 위의 지표 확인


# In[84]:


# end of file

