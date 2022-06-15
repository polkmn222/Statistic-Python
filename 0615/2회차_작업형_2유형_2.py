#!/usr/bin/env python
# coding: utf-8

# In[19]:


import pandas as pd
import numpy as np

X_train = pd.read_csv('./datasets/datasets/part3/204_x_train.csv')
y_train  = pd.read_csv('./datasets/datasets/part3/204_y_train.csv')
X_test = pd.read_csv('./datasets/datasets/part3/204_x_test.csv')


# In[20]:


print('X_train의 shape:', X_train.shape)
print('X_test의 shape:', X_test.shape)
print('y_train의 shape:', y_train.shape)

### 아래의 셀에서 X_train + X_test = X_all로 만들 예정


# In[21]:


## pd.concat 활용
X_all = pd.concat([X_train, X_test])

### 데이터 정보 확인
X_all.info()

### object인 자료의 컬럼들만 호출
print('\n문자형 자료의 컬럼:\n', X_all.select_dtypes(include='object').columns)


# In[31]:


# 1번풀이에서 정확도가 높지 않았으므로 우리는 X변수들을 
# 재확인할 필요가 있습니다.

print(X_train.columns)

## ID -식별자라 제외

# Warehouse_block - 몇개의 범주로만 구성되어 있으므로
# 범주형 자료 - multi class로 봐도 무방
print('\nWarehouse_block:\n', X_train.Warehouse_block.value_counts())

# Mode_of_Shipment - 범주형
# multi_class
print('\nMode_of_Shipment:\n', X_train.Mode_of_Shipment.value_counts())

# Mode_of_Shipment - 범주형
# multi_class
print('\nCustomer_care_calls:\n', X_train.Customer_care_calls.value_counts())

# Customer_rating - 범주형
# multi_class
print('\nCustomer_rating:\n', X_train.Customer_rating.value_counts())

# Cost_of_the_Product - 연속형
# multi_class
print('\nCost_of_the_Product:\n', X_train.Cost_of_the_Product.value_counts())

# Prior_purchases - 범주형
# multi_class
print('\nPrior_purchases:\n', X_train.Prior_purchases.value_counts())

# Product_importance - 범주형
# multi_class
print('\nProduct_importance:\n', X_train.Product_importance.value_counts())

# Gender - 범주형
# multi_class
print('\nGender:\n', X_train.Gender.value_counts())

# Discount_offered - 연속형
# multi_class
print('\nDiscount_offered:\n', X_train.Discount_offered.value_counts())

# Weight_in_gms - 연속형
# multi_class
print('\nWeight_in_gms:\n', X_train.Weight_in_gms.value_counts())


# In[36]:


X_all_contig = X_all[['Cost_of_the_Product','Discount_offered','Weight_in_gms']]

## X_train과 X_all의 연속형 변수들만으로 구성한 후 split
X_train_fin2 = X_all_contig[:6599]
X_test_fin2 = X_all_contig[6599:]


## 전처리 대상이 아니므로 y_train_fin을 카피하여 사용
y_train_fin2 = y_train_fin.copy()


# In[39]:


### 학습을 수행하기 위해 train_test_split을 활용한 val data를 생성
from sklearn.model_selection import train_test_split

xtrain2, xval2, ytrain2, yval2 = train_test_split(X_train_fin2, y_train_fin2,
                                             test_size=0.2,
                                             stratify= y_train_fin2,
                                             random_state=615)


# In[40]:


## 학습할 알고리즘 호출
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from lightgbm import LGBMClassifier

from sklearn.metrics import accuracy_score, roc_auc_score

### rf_clf를 적용
rf_clf = RandomForestClassifier(random_state=615)
rf_clf.fit(xtrain2,ytrain2)
pred_rf = rf_clf.predict(xval2)

accuracy_rf = accuracy_score(yval2,pred_rf)
roc_auc_rf = roc_auc_score(yval2,pred_rf)

## 평가지표 적용
print('rf_clf의 정확도:', np.round(accuracy_rf,4))
print('rf_clf의 roc_auc점수:', np.round(roc_auc_rf,4))

### gb_clf를 적용
gb_clf = GradientBoostingClassifier(random_state=615)
gb_clf.fit(xtrain2,ytrain2)
pred_gb = gb_clf.predict(xval2)

accuracy_gb = accuracy_score(yval2,pred_gb)
roc_auc_gb = roc_auc_score(yval2,pred_gb)

## 평가지표 적용
print('gb_clf의 정확도:', np.round(accuracy_gb,4))
print('gb_clf의 roc_auc점수:', np.round(roc_auc_gb,4))


# In[ ]:


### test1 번 종료를 하겠습니다.


# In[44]:


X_all.head(3)


# In[48]:


### test2번 
X_all_contig = X_all[['Customer_care_calls','Warehouse_block','Prior_purchases','Mode_of_Shipment','Cost_of_the_Product','Discount_offered','Weight_in_gms']]
X_all_contig

### 원-핫 인코딩을 적용해보자...
X_all_oh_contig = pd.get_dummies(X_all_contig)


# In[49]:


## X_train과 X_all의 연속형 변수들만으로 구성한 후 split
X_train_fin3 = X_all_oh_contig[:6599]
X_test_fin3 = X_all_oh_contig[6599:]


## 전처리 대상이 아니므로 y_train_fin을 카피하여 사용
y_train_fin3 = y_train_fin.copy()


# In[50]:


### 학습을 수행하기 위해 train_test_split을 활용한 val data를 생성
from sklearn.model_selection import train_test_split

xtrain3, xval3, ytrain3, yval3 = train_test_split(X_train_fin3, y_train_fin3,
                                             test_size=0.2,
                                             stratify= y_train_fin3,
                                             random_state=615)


# In[51]:


## 학습할 알고리즘 호출
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from lightgbm import LGBMClassifier

from sklearn.metrics import accuracy_score, roc_auc_score

### rf_clf를 적용
rf_clf = RandomForestClassifier(random_state=615)
rf_clf.fit(xtrain3,ytrain3)
pred_rf = rf_clf.predict(xval3)

accuracy_rf = accuracy_score(yval3,pred_rf)
roc_auc_rf = roc_auc_score(yval3,pred_rf)

## 평가지표 적용
print('rf_clf의 정확도:', np.round(accuracy_rf,4))
print('rf_clf의 roc_auc점수:', np.round(roc_auc_rf,4))

### gb_clf를 적용
gb_clf = GradientBoostingClassifier(random_state=615)
gb_clf.fit(xtrain3,ytrain3)
pred_gb = gb_clf.predict(xval3)

accuracy_gb = accuracy_score(yval3,pred_gb)
roc_auc_gb = roc_auc_score(yval3,pred_gb)

## 평가지표 적용
print('gb_clf의 정확도:', np.round(accuracy_gb,4))
print('gb_clf의 roc_auc점수:', np.round(roc_auc_gb,4))


# In[52]:


from lightgbm import LGBMClassifier

lgbm_wrapper = LGBMClassifier(n_estimators = 400)

# LightGBM early_stopping_rounds 확인
evals3 = [(xval3,yval3)]
lgbm_wrapper.fit(xtrain3,ytrain3,early_stopping_rounds=100,
                eval_metric='logloss',
                eval_set = evals3,
                verbose=True)

pred_lgbm3 = lgbm_wrapper.predict(xval3)

accuracy_lgbm = accuracy_score(yval3,pred_lgbm3)
roc_auc_lgbm = roc_auc_score(yval3,pred_lgbm3)

print('lgbm_clf의 정확도:', np.round(accuracy_lgbm,4))
print('lgbm_clf의 roc_auc_점수:', np.round(roc_auc_lgbm,4))


# In[32]:


## test_coding에서도 즉, one-hot 인코딩을 적용해보아도 더 높은
## accuracy 및 roc_auc 점수를 devoloping(개선)시키지 못하였습니다.

# 혹시라도 여러분들께서 더 좋은 점수가 나오신다면
# 같이 공유 부탁드립니다.

# 오전 수업 수고많으셨습니다 ^^


# In[5]:


# #### Label Encoding시 분석 알고리즘

# ### 불필요속성

# X_all_drop = X_all.drop(['ID'], axis=1)


# ### MinMaxScaler를 적용해본다.
# ### MinMaxScaling을 적용하는 이유는 간단한데,
# ### 데이터들의 정규성 가정을 우리가 확신하지 못해서입니다.

# from sklearn.preprocessing import MinMaxScaler

# mm_scaler = MinMaxScaler()
# result_ndarray = mm_scaler.fit_transform(X_all_drop)

# print(result_ndarray)

# ## 전처리된 X값들과 y값을 재정의
# X_train_fin = result_ndarray[:6599]
# X_test_fin = result_ndarray[6599:]
# y_train_fin = y_train['Reached.on.Time_Y.N']


# In[6]:


# ### 학습을 수행하기 위해 train_test_split을 활용한 val data를 생성
# from sklearn.model_selection import train_test_split

# xtrain, xval, ytrain, yval = train_test_split(X_train_fin, y_train_fin,
#                                              test_size=0.2,
#                                              stratify=y_train_fin,
#                                              random_state=615)


# In[7]:


# ## 학습할 알고리즘 호출
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.ensemble import GradientBoostingClassifier
# from lightgbm import LGBMClassifier

# from sklearn.metrics import accuracy_score, roc_auc_score

# ### rf_clf를 적용
# rf_clf = RandomForestClassifier(random_state=615)
# rf_clf.fit(xtrain,ytrain)
# pred_rf = rf_clf.predict(xval)

# accuracy_rf = accuracy_score(yval,pred_rf)
# roc_auc_rf = roc_auc_score(yval,pred_rf)

# ## 평가지표 적용
# print('rf_clf의 정확도:', np.round(accuracy_rf,4))
# print('rf_clf의 roc_auc점수:', np.round(roc_auc_rf,4))

# ### gb_clf를 적용
# gb_clf = GradientBoostingClassifier(random_state=615)
# gb_clf.fit(xtrain,ytrain)
# pred_gb = gb_clf.predict(xval)

# accuracy_gb = accuracy_score(yval,pred_gb)
# roc_auc_gb = roc_auc_score(yval,pred_gb)

# ## 평가지표 적용
# print('gb_clf의 정확도:', np.round(accuracy_gb,4))
# print('gb_clf의 roc_auc점수:', np.round(roc_auc_gb,4))


# In[8]:


# ### lightgbm 적용

# from lightgbm import LGBMClassifier

# # 400개의 분류기를 생성
# lgbm_wrapper = LGBMClassifier(n_estimators=400)

# evals = [(xval,yval)]

# lgbm_wrapper.fit(xtrain,ytrain, early_stopping_rounds=100,
#                 eval_metric='logloss',
#                 eval_set = evals,
#                 verbose = False) # 미리 정확도가 아닌 'logloss'지표로
#                                 # 알고리즘의 over-fitting방지 및
#                                 # 정확도를 살펴본다.

# pred_lgbm = lgbm_wrapper.predict(xval) # xval을 넣었을때 예상되는 y값

# accuracy_lgbm = accuracy_score(yval,pred_lgbm)
# roc_auc_lgbm = roc_auc_score(yval,pred_lgbm)

# ## 평가지표 적용
# print('lgbm_clf의 정확도:', np.round(accuracy_lgbm,4))
# print('lgbm_clf의 roc_auc점수:', np.round(roc_auc_lgbm,4))


# In[9]:


# ## 최종결과 제출 코드 ##

# final_model = gb_clf.fit(X_train_fin, y_train_fin)
# y_pred = final_model.predict(X_test_fin)

# ### 제출파일을 생성
# submit_df = pd.DataFrame({'y_pred':y_pred}).reset_index()
# submit_df

# submit_df.to_csv('./220615.csv')


# In[53]:


# end of file -2 

