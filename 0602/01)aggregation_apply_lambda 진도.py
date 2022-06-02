#!/usr/bin/env python
# coding: utf-8

# ## 정렬, Aggreagation 함수, GroupBy 적용

# DataFrame과 Series의 정렬을 위해서는 sort_values( ) 메서드를 이용한다. sort_values()는 RDBMS SQL의 order by키워드와 유사하다. sort_values()의 주요 입력 파라미터는 by, ascending, inplace이다. by로 특정 칼럼을 입력하면 해당 칼럼으로 정렬을 수행한다. ascending=True로 설정하면 오름차순으로 정렬하며, ascending=False로 설정하면 내림차순으로 설정된다. 기본(default)는 ascendling=True이다. inplace도 default는 inplace=False이다.

# In[46]:


import numpy as np
import pandas as pd


# In[47]:


titan_df = pd.read_csv('./train.csv')

## Series의 경우에는 Value들이 값으로 호출되어
## sort_values를 그냥 사용하여도 됨
titan_df['Name'].sort_values()


# In[48]:


### Dataframe을 정렬할 경우

titan_df.sort_values(by = 'Age')  # 기준이 될 하나의 컬럼을 잡아서 정렬


# In[49]:


## 내림차순 정렬
# Ascending parameter를 조절
titan_df.sort_values(by = 'Age', ascending=False)


# In[50]:


## 1차 오름차순, 2차 오름차순??
titan_df.sort_values(by=['Name', 'Age'], ascending=[True, False]).head(2)


# ### Aggregation 함수 적용

# Aggregaion :: min(), max(), sum(), count()와 같이 총합 또는 총계처리를 위한 함수

# In[51]:


titan_df.count()


# In[52]:


### 특정 컬럼들의 평균 및 aggregation을 보고 싶다.
titan_df[['Age', 'Fare']].mean()


# ### groupby() 적용

# In[53]:


titan_df['Sex'].value_counts()


# In[54]:


titan_df.groupby(by='Sex').count()


# In[55]:


# 결촉값이 존재하는
# 'Age'와 'Cabin'의 컬럼값만 추출 - 1
titan_df.groupby(by='Pclass').count()[['Age', 'Cabin']]


# In[56]:


# 'Age'와 'Cabin'의 컬럼값만 추출 - 2
titan_df.groupby(by='Pclass')[['Age', 'Cabin']].count()


# In[57]:


# 'Pclass'에 따른 'Age', 'Cabin'의 최소, 최대값을
# groupby를 통해 구해보자.

titan_df.groupby(by='Pclass')[['Age', 'Fare']].agg(['max', 'min'])


# 서로 다른 aggregation 함수를 groupby에서 호출하려면
# SQL은 Select max(Age), sum(SibSp), avg(Fare) from titanic_table group by Pclass와 같이 쉽게 가능하다.

# In[58]:


agg_format = {'Age' : 'max',
              'SibSp' : 'sum',
              'Fare' : 'mean'}
titan_df.groupby(by='Pclass').agg(agg_format)


# ### fillna()로 결측값 데이터 대체

# In[59]:


titan_df.isna()


# In[60]:


### 결측값의 개수 확인
titan_df.isna().sum()


# In[61]:


titan_df1 = titan_df['Cabin'].fillna('N')
titan_df1


# In[62]:


titan_df['Cabin'] = titan_df['Cabin'].fillna('N')  ## reassign(재할당)
titan_df['Age'] = titan_df['Age'].fillna(np.mean(titan_df['Age']))
titan_df['Embarked'] = titan_df['Embarked'].fillna('N')

titan_df.isna().sum()


# ### apply lambda 식으로 데이터 가공

# In[63]:


### 사용자 정의 함수
def get_square(x):
    result = x ** 2
    return result


# In[64]:


print('3의 제곱', get_square(3))


# In[65]:


lambda_square = lambda x: x**2


# In[66]:


print('lambda로 만든 3의 제곱', lambda_square(3))


# In[67]:


a = [1, 2, 3]

result = map(lambda  x : x ** 2, a)
list(result)


# In[68]:


### titan_df의 Name컬럼을 통해
### 각 관측치(obs :: row)의 이름의 길이를 구해봅시다.
###


# In[69]:


titan_df['Name'].apply(lambda x: len(x))


# In[70]:


titan_df['Na_len'] = titan_df['Name'].str[:].apply(lambda x : len(x))

### apply lamda를 모른다면?
### for문으로 풀이가 가능은 하다.

### 위의 결과를 for문으로 한 번 해보셔요 ^^
result = []
for i in range(len(titan_df['Name'])):
    result.append(len(titan_df.loc[i, 'Name']))


# In[71]:


titan_df['test'] = result   # test명의 컬럼을 생성
titan_df.head()


# In[72]:


### Age 컬럼으로 young과 adult를 구분해보자
titan_df['Age_cat'] = titan_df['Age'].apply(lambda x: 'young' if x < 30 else ('adult' if x < 60 else 'elderly'))
titan_df.Age_cat.value_counts()


# In[74]:


def add(first, last):
    return first + last


# In[75]:


add(3, 4)


# In[77]:


### age_cat을 재정의하는 함수를 만들어보자.
def get_cat(x):
    char = ''
    if x <= 5: char = 'baby'
    elif x<= 12: char = 'child'
    elif x<= 18: char = 'teen'
    elif x<= 25: char = 'student'
    elif x<= 45: char = 'young_adult'
    elif x<= 60: char = 'adult'
    else: char = 'elderly'

    return char


# In[79]:


titan_df['Age_cat'] = titan_df.Age.apply(lambda x: get_cat(x))
titan_df


# In[80]:


# end of file

