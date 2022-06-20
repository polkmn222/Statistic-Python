#!/usr/bin/env python
# coding: utf-8

# ### LinearRegression 클래스 - Ordinary Least Squares 

# * CRIM : 지역별 범죄 발생률
# * ZN : 25,000평방피트를 초과하는 거주 지역의 비율
# * INDUS : 비상업 지역 넓이 비율
# * CHAS : 찰스강에 대한 더미변수(강의 경계에 위치한 경우 1, 아니면 0)
# * NOX : 일산화질소 농도
# * RM : 거주할 주거의 방 개수 
# * AGE : 건축된 소유 주택의 연식, 1940년 이전에 건축된 소유주택 
# * DIS : 5개 주요 고용센터까지의 가중 거리
# * RAD : 고속도로 접근 용이도
# * TAX : 10,000달러당 책정된 재산세율
# * PTRATIO : 지역의 교사와 학생 수 비율
# * B : 지역의 흑인 거주 비율
# * LSTAT : 하위 계층의 비율
# * MEDV : 본인 소유 주택 가격에서의 중앙값(Median)

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from sklearn.datasets import load_boston
get_ipython().run_line_magic('matplotlib', 'inline')

# boston 데이터 세트 로드
boston = load_boston()
boston


# In[8]:


## boston datasets 변환
boston_df = pd.DataFrame(boston.data, columns = boston.feature_names)

## boston_df의 y값을 생성
boston_df['PRICE'] = boston.target

print('boston_df의 shape:', boston_df.shape)
boston_df.head(4)


# In[19]:


boston_df.columns


# In[53]:


### 선형성을 보기 위한 plot 그리기

fig, axs = plt.subplots(figsize=(16,8), ncols=4, nrows=2)
lm_ftrs=['CRIM', 'ZN', 'INDUS', 'NOX', 'RM', 'PTRATIO', 'B', 'LSTAT']
len(lm_ftrs)

for i, ftr in enumerate(lm_ftrs):
    row = int(i/4) # 몫을 int로 가져오게 함
    col = i%4 # 나머지를 가져오게 함
    sns.regplot(x=ftr, y='PRICE', data=boston_df, ax=axs[row,col])

# ### NGD(노가다)기법
# # 0번째 행
# sns.regplot(x='CRIM', y='PRICE', data=boston_df, ax=axs[0][0])
# sns.regplot(x='ZN', y='PRICE', data=boston_df, ax=axs[0][1])
# sns.regplot(x='INDUS', y='PRICE', data=boston_df, ax=axs[0][2])
# sns.regplot(x='NOX', y='PRICE', data=boston_df, ax=axs[0][3])

# # 1번째 행
# sns.regplot(x='RM', y='PRICE', data=boston_df, ax=axs[1][0])
# sns.regplot(x='PTRATIO', y='PRICE', data=boston_df, ax=axs[1][1])
# sns.regplot(x='B', y='PRICE', data=boston_df, ax=axs[1][2])
# sns.regplot(x='LSTAT', y='PRICE', data=boston_df, ax=axs[1][3])


# In[59]:


### 간단하게 선형회귀 분석을 수행해본다.
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

X_data = boston_df.drop(['PRICE'], axis=1)
y_target = boston_df['PRICE']

print(X_data.shape)
print(y_target.shape)

# train_test_spit을 수행할 예정 :: test_size = 0.3, random_state =156
X_train, X_test, y_train, y_test = train_test_split(X_data, y_target,
                                                   test_size= 0.3,
                                                   random_state=156)


# In[63]:


### train, test의 shape을 확인
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

## 선형회귀 모델로 fit/pred/eval

lr = LinearRegression()
lr.fit(X_train, y_train) # fitting을 시킴

preds_lr = lr.predict(X_test)

## 지표를 생성 및 적용
mse = mean_squared_error(y_test,preds_lr)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test,preds_lr)

r2_1 = r2_score(y_test, preds_lr)

print('MSE:{0:.4f}, RMSE:{1:.4f}, MAE:{2:.4f}'.format(mse, rmse, mae))


# In[67]:


boston_df.PRICE.describe()


# In[70]:


# y = w1 * x1 + w2 * x2 + b ... 
# intercept(절편)과 coefficient(계수)값

print('절편 값:', np.round(lr.intercept_,4))
print('회귀계수값:', np.round(lr.coef_,4))

# 회귀계수를 소수점 2째자리까지 맞춰줌
coef_1 = np.round(lr.coef_,2)


# In[74]:


# 회귀계수 정렬
coef_sr = pd.Series(data=coef_1, index= X_data.columns)
coef_sr.sort_values()


# In[77]:


from sklearn.model_selection import cross_val_score

y_target = boston_df['PRICE']
X_data = boston_df.drop(['PRICE'], axis=1)
lr_reg = LinearRegression()

# cross_val_score()로 fold 5개로 set
# MSE를 구한뒤
# 이를 기반으로 RMSE

# neg_mse_scores는 mean_squared_error에 -1이 곱해진 상태
neg_mse_scores = cross_val_score(lr_reg, X_data, y_target, scoring='neg_mean_squared_error', cv=5)
rmse_scores = np.sqrt(-1*neg_mse_scores)
avg_rmse = np.mean(rmse_scores)

# Negative MSE scores는 전부 다 음수
print('5 folds의 개별 Neg MSE scores:', np.round(neg_mse_scores,2))
print('5 folds의 개별 RMSE scores:', np.round(rmse_scores,2))
print('5 folds의 평균 RMSE scores:', np.round(avg_rmse,2))


# ## 다항 회귀를 통한 under-fitting and over-fitting

# In[114]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline # 첫 출
from sklearn.preprocessing import PolynomialFeatures # 다항회귀 :: 곡선결과
from sklearn.linear_model import LinearRegression # 단순선형회귀
from sklearn.model_selection import cross_val_score # 교차검증
from sklearn.metrics import SCORERS


# In[79]:


# 임의의 값으로 구성된 X값에 대해
# 코사인 변환 값을 반환.

def true_func(X):
    return np.cos(1.5 * np.pi * X)


# In[94]:


# X는 0~ 1까지의 30개의 임의의 값을
# 샘플링을 해보자.
np.random.seed(0)
n_samples=30 
X = np.sort(np.random.rand(n_samples))
X

# y값은 코사인 기반의 true_func
# 약간의 노이즈 변동을 더해서

y = true_func(X) + np.random.randn(n_samples) * 0.1
y


# In[120]:


# 다항식 차수를 각각 1, 4, 15로 변경해서 결과를 비교하기
plt.figure(figsize=(14,5))
degree = [1,4,15]

# 다항 회귀의 차수(degree)를 1,4,15로 각각
# 변화시키며 변화를 비교
for i in range(len(degree)):
    ax = plt.subplot(1,len(degree), i+1)
    plt.setp(ax, xticks=(), yticks=())
    
    ### 개별 degree 별로 Polynomial(다항식) 변환
    polynomial_features  = PolynomialFeatures(degree= degree[i], include_bias= False)
    linear_regression = LinearRegression()
    pipeline = Pipeline([("poly_ftrs", polynomial_features),
                         ("linear_reg", linear_regression)])
    X_train = X.reshape(-1,1)    
    
    # pipeline의 학습
    pipeline.fit(X_train, y) # X값을 2차원으로 만든다. 
    
    # 교차검증으로 다항회귀를 평가 
    scores = cross_val_score(pipeline, X_train, y, scoring='neg_mean_squared_error', cv=10)
    
    # pipeline을 구성하는 세부 객체를 named_steps['객체명']을 활용해
    # 회귀계수를 도출
    coefficients = pipeline.named_steps['linear_reg'].coef_
    print('\nDegree {0} 회귀 계수는 {1}입니다.'.format(degree[i], np.round(coefficients,2)))
    print('\nDegree {0} MSE는 {1}입니다.'.format(degree[i], -1*np.mean(scores)))
    
     # 0부터 1까지 테스트 데이터 세트를 100개로 나눠 예측을 수행
    # 테스트 데이터 세트에 회귀 예측을 수행 및 예측 곡선과 실제 곡선을 그려서 비교
    X_test = np.linspace(0,1,100)
    # 예측값 곡선
    plt.plot(X_test, pipeline.predict(X_test[:,np.newaxis]), label="Model")
    # 실제값 곡선
    plt.plot(X_test, true_func(X_test), '--', label="True function")
    plt.scatter(X, y, edgecolors='b', s=20, label="Samples")
    
    plt.xlabel("x");plt.ylabel("y");plt.xlim((0,1));plt.ylim((-2,2)); plt.legend(loc='best')
    plt.title("Degree {}\n MSE={:.2e}(+/-{:.2e})".format(degrees[i], -scores.mean(), scores.std()))
    
plt.show()


# In[121]:


# end of files

