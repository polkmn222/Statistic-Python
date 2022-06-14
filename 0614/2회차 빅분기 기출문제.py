#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# 01 다음은 Boston Housing 데이터 세트이다.
# 범죄율 컬럼인 CRIM 항목의 상위에서 10번째 값
# (즉, 범죄율을 큰 순서대로 오름차순 정렬했을 때 10번째에 위치한 값)으로
# 상위 10개의 값을 변환한 후,
# AGE가 80이상인 데이터를 추출하여 CRIM의 평균값을 계산하시오.


# In[42]:


import pandas as pd
import numpy as np

# 데이터 로딩
boston_df = pd.read_csv('./Part3/201_boston.csv')


# In[50]:


# 데이터 정렬 및 10번째 값 호출
sorted_df = boston_df.sort_values(by='CRIM', ascending=False)
sorted_df[:10].shape
values_10th = sorted_df.CRIM.iloc[9]
print('정렬된 데이터:\n', sorted_df.head(10))
print('\n10번째의 값:', values_10th)


# In[63]:


## 10번째의 값을 -> 0~9번까지로 대입
sorted_df['CRIM'][:10] = values_10th  # version1
sorted_df.iloc[:10,:0] = values_10th  # version2
sorted_df['CRIM'][:15]

## 조건식 :: Age > 80
cond1 = sorted_df[sorted_df.AGE > 80]
Answer = cond1['CRIM'].mean()
print(Answer)


# In[ ]:


# 02 다음은 California Housing 데이터 세트이다.
# 주어진 데이터의 첫 번째 행부터 순서대로 80%까지의 데이터를 훈련데이터로
# 추출한 후, 전체 방 개수 컬럼을 의미하는
#‘total bedrooms’변수의 결측치를
#‘total_bedrooms’변수의 중앙값으로 대체한
# 데이터 세트를 구성한다.


# 결측치 대체 전의 ‘total_bedrooms’변수 표준편차 값과
# 결측치 대체 후의 ‘total_bedrooms’변수 표준편차 값의 차이에 대한
# 절대값을 계산하세요.


# In[78]:


import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

# 데이터 로딩
cal_df = pd.read_csv('./Part3/202_housing.csv')
cal_df_80 = len(cal_df) * 0.8

train_df = cal_df[:int(cal_df_80)]

before_std = train_df.total_bedrooms.std()
print('결측치 대체 전의 표준편차:', before_std)


# In[79]:


### 결측치 대체
train_md = train_df.total_bedrooms.median()
train_md

train_df['total_bedrooms_imp'] = train_df.total_bedrooms.fillna(train_md)

after_std = train_df.total_bedrooms_imp.std()
print('결측치 대체 후의 표준편차:', after_std)
print('절대값 계산', np.abs(before_std - after_std))


# In[ ]:


# 03 2번 문항에서 활용한 California Housing 데이터 세트를 그대로 활용한다.
# 인구 컬럼인 population 항목의 이상값의 합계를 계산하시오.
# (※ 이상값은 사분위수에서 1.5 X 표준편차를 초과하거나 미만인 값의 범위로 정한다.)


# In[84]:


import pandas as pd
import numpy as np

# 데이터 로딩
cal_df = pd.read_csv('./Part3/202_housing.csv')

### cal_df.population의 표준편차는?
pop_std = cal_df.population.std()
strandard_out = pop_std * 1.5

### 1사분위수에서 뺀 값 vs 3사분위수에서는 더할 값
lower_q1 = np.quantile(cal_df.population, 0.25) - strandard_out
upper_q3 = np.quantile(cal_df.population, 0.75) + strandard_out

### 이상치값의 합계를 구하기 위한 조건식
result_cond = cal_df[(cal_df.population < lower_q1) | (cal_df.population > upper_q3)]
result_cond['population'].sum()


# In[ ]:


# end of file

