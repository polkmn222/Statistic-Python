#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd

body_df = pd.read_csv('./body.csv')


# In[8]:


# Q1. 전체데이터의 수축기혈압(최고) -  이완기혈압(최저)의 평균을 구해보세요.


# In[24]:


result = (body_df['수축기혈압(최고) : mmHg']-body_df['이완기혈압(최저) : mmHg']).mean()
print(result)


# In[9]:


# Q2. 50~59세의 신장평균을 구해보세요 


# In[32]:


average_height = body_df[(body_df['측정나이']<60)&(body_df['측정나이']>=50)].iloc[:,3].mean()
print(average_height)


# In[33]:


# Q3. 연령대 (20~29:20대 
#              30~39: 30대)등 각 연령대별 인원수를 구해보세요 


# In[38]:


body_df['연령대'] = body_df.측정나이 //10 * 10 
body_df['연령대'].value_counts()


# In[39]:


# Q4. 남성 중 A등급과 D등급의 체지방률 평균의 차이(큰 값에서 작은 값의 차)를 구해보세요.


# In[46]:


import numpy as np


A_grade = body_df[(body_df.측정회원성별 == 'M') & (body_df.등급 == 'A')].iloc[:,5].mean()
D_grade = body_df[(body_df.측정회원성별 == 'M') & (body_df.등급 == 'D')].iloc[:,5].mean()

np.abs(A_grade - D_grade)


# In[12]:


# Q5. 여성 중 A등급과 D등급의 체지방률 평균의 차이(큰 값에서 작은 값의 차)를 구해보세요.


# In[47]:


import numpy as np


A_grade = body_df[(body_df.측정회원성별 == 'F') & (body_df.등급 == 'A')].iloc[:,5].mean()
D_grade = body_df[(body_df.측정회원성별 == 'F') & (body_df.등급 == 'D')].iloc[:,5].mean()

np.abs(A_grade - D_grade)


# In[13]:


# Q6 bmi는 자신의 몸무게(kg)를 키의 제곱(m)으로 나눈 값입니다. 데이터의 bmi를 구한 새로운
# 컬럼을 만들고 남성과 여성의 bmi 평균을 구해보세요.


# In[62]:


height_squared = (body_df['신장 : cm']/100)**2 # m 단위므로 cm를 /100으로 나누어 줍니다.
bmi = body_df['체중 : kg']/height_squared

body_df['bmi'] = bmi

male_average = body_df[body_df['측정회원성별'] == 'M'].bmi.mean()
female_average = body_df[body_df['측정회원성별'] == 'F'].bmi.mean()

print('남성 평균:', male_average)
print('여성 평균:', female_average)


# In[14]:


# Q7 bmi보다 체지방률이 높은 사람들의 체중 평균을 구해보세요.


# In[68]:


answer = body_df[(body_df['bmi']<body_df['체중 : kg'])]['체중 : kg'].mean()
print(answer)


# In[15]:


# Q8 남성과 여성의 악력 평균의 차이를 구해보세요.


# In[76]:


import numpy as np
import pandas as pd

male_average_grip = body_df[body_df.측정회원성별 == 'M']['악력D : kg'].mean()
female_average_grip = body_df[body_df.측정회원성별 == 'F']['악력D : kg'].mean()

np.abs(male_average_grip -  female_average_grip)

### 또는

result = body_df.groupby('측정회원성별')['악력D : kg'].mean()
np.abs(result.M - result.F)


# In[16]:


# Q9 남성과 여성의 교차 윗몸일으키기 횟수의 평균의 차이를 구해보세요. 


# In[77]:


result1 = body_df.groupby('측정회원성별')['교차윗몸일으키기 : 회'].mean()
np.abs(result1.M - result1.F)


# In[78]:


# end of file

