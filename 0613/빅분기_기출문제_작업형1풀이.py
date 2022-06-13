#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 01 다음은 California Housing 데이터 세트이다. 데이터 중 결측치가 있는 경우
# 해당 데이터의 행을 모두 제거하고, 
# hint :: dropna라고하는 함수가 활용


# 첫 번째 행부터 순서대로 70%까지의 데이터를
# 훈련데이터로 추출한 데이터 세트를 구성한다.
# 슬라이싱으로 쓰시되 슬라이싱은 int만 가능합니다.


# 변수 중 ‘housing_median_age’의
# Q1(제1사분위수)값을 정수로 계산하시오. 
# quantile # pandas
# np.quantile # numpy


# In[2]:


import pandas as pd
import numpy as np
df = pd.read_csv('./datasets/datasets/part3/301_housing.csv')
df.info()

# 결측치 drop이 된 함수
housing_drop_na = df.dropna(inplace=False)

## 원본 data와 결측치 제거 후의 data를 비교한다.
print('\n ### 원본 data의 수 :', df.shape[0])
print('\n ### 결측치 제거 후의 data의 수 :', len(housing_drop_na))

## housing_drop_na의 70%를 훈련데이터로 설정한다.
train_data = housing_drop_na[:int(len(housing_drop_na)*0.7)]


# # 변수 중 ‘housing_median_age’의
# # Q1(제1사분위수)값을 정수로 계산하시오. 
# quantile # pandas
# np.quantile # numpy

Answer = np.quantile(train_data.housing_median_age, 0.25)
print('1번문제의 답은:',Answer)


# In[3]:


# 02 다음은 국가별 연도별 인구 10만 명당 결핵 유병률 데이터 세트이다. 
# 2000년도의 국가별 결핵 유병률 데이터 세트에서 
# 2000년도의 평균값보다 더 큰 유병률 값을 가진 국가의 수를 계산하시오.

import pandas as pd
import numpy as np
df = pd.read_csv('./datasets/datasets/part3/302_worlddata.csv')
df_t = df.transpose()

## 전치행렬을 한 이후에 컬럼명 변경
df_t.rename(columns = {0:'1999',
                      1:'2000',
                      2:'2001',
                      3:'2002'}, inplace=True)

df_t = df_t.drop('year', axis=0) # reassignment
df_t ## 전처리 끝

## condition에 해당하는 df호출

len(df_t[df_t['2000']>np.mean(df_t['2000'])])


# In[4]:


### 2000년도 데이터
"""
java style

cond1 = disease_df['year']==2000
disease_df_2000 = disease_df[cond1]
print(disease_df_2000)

### 2000년도 평균값
Sum = 0
cnt = 0
for _col in range(1,194):
    Sum = Sum + disease_df_2000.iloc[0,_col]
    cnt = cnt+1
    
mean = Sum/cnt
print(mean)

### 국가의 수 계산
nation_cnt = 0
for _col in range(1,194):
    if disease_df_2000.iloc[0,_col] > mean:
        nation_cnt = nation_cnt+1

print(nation_cnt)
"""


# In[5]:


# 03 다음은 Titanic 데이터 세트이다. 주어진 데이터 세트의 컬럼 중 빈 값 또는
# 결측치를 확인하여, 결측치의 비율이 가장 높은 변수명을 출력하시오.

import pandas as pd
import numpy as np
titan_df = pd.read_csv('./datasets/datasets/part3/303_titanic.csv')

# 가장 결측값의 비율이 큰 컬럼을 info를 확인
titan_df.info()

# 코드로 답안 작성
cond1= titan_df.isna().sum()

# version-1
answer = cond1.index[5] 

# version-2
answer2 = cond1.index[cond1.argmax()] 
print(answer2)


# In[6]:


# end of files

