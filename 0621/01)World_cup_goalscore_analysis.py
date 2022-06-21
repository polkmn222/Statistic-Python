#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np

goal_df = pd.read_csv('./worldcupgoals.csv')
goal_df


# In[5]:


# Q1. 전체기간에서의 각 나라별 골득점수가 가장 높은 상위 5개의 나라와 그 득점수를
# df형태로 출력해보자.


# In[6]:


goal_df.groupby(by='Country').sum().sort_values('Goals',ascending=False).head(5)


# In[7]:


# Q2. 전체기간에서의 각 나라별 '골을 득점한 선수'가 가장 높은 상위 5개의 나라와 그 득점수를
# df형태로 출력해보자.


# In[8]:


## 풀이 :: 각 나라별로 출력을 해야하다 보니 나라명을 기준으로 groupby를 진행합니다.
##         또한 각 나라별로 모였던 것에서 위의 문제는 Goal득점수를 풀이하다보니
##         더한 것이고, 현재의 문제에서는 득점한 선수(하나의 개체를 하나씩 세는 행위)
##         이므로 size함수를 활용한 것입니다.

goal_df.groupby(by='Country').size().sort_values(ascending=False).head(5)


# In[9]:


### Q3. Years의 컬럼이 년도의 형식이 좀 다릅니다. 그러므로 각 형식과 다른 값들을 한 번 살펴봅니다.


# In[10]:


goal_df['len_year'] = goal_df.Years.str.split('-')
goal_df

## 사용자 정의 함수 적용
def check_str(df_len_year):
    for value in df_len_year:
        if len(str(value)) != 4:
            return False
        else: 
            return True

goal_df


# In[11]:


result = goal_df['len_year'].apply(check_str).value_counts()

print('흠결 데이터의 확인\n', result)

## 만약 45개의 데이터만 나오게 하고 싶다면
goal_df['len_check'] = goal_df['len_year'].apply(check_str)
len(goal_df[goal_df.len_check!=False])


# In[12]:


### Q4. 3번문제에서 정의된 케이스들을 제외한 df를 정의하되 변수명을 goal_df2라 하자. 
### 그리고 데이터의 shape및 수를 출력해보세요.


# In[13]:


goal_df2 = goal_df[goal_df.len_check!=False].copy()
len(goal_df2)


# In[16]:


### Q5. 월드컵 출전횟수를 나타내는 'NO.cup' 컬럼을 추가하고 4회 출전한 선수의 수를 구하세요.


# In[28]:


goal_df2['NO_cup'] = goal_df2['len_year'].str.len()
goal_df2[goal_df2['NO_cup']==4].shape[0]

## 위의 방법과는 다를 수도 있다
## 그냥 NO_cup은 출전연도의 데이터를 변화시킨 것이므로
goal_df2.NO_cup.value_counts()[4]  ## 여기서 4는 인덱스를 의미합니다


# In[17]:


### Q6. South Korea 국가의 월드컵의 출전횟수가 4회인 선수들의 숫자를 구해보세요.


# In[29]:


goal_df2[(goal_df2['Country']=='South Korea') & (goal_df2['NO_cup'] >= 4)]


# In[18]:


### Q7. 2002 월드컵에 출전한 전체 선수의 수는 몇 명입니까?


# In[30]:


len(goal_df2[goal_df2.Years.str.contains('2002')])


# In[19]:


### Q8. 이름에 'choi'단어가 들어가는 선수는 몇멍인가요? (대, 소문자는 구분하지 않습니다.)


# In[31]:


goal_df2[goal_df2.Player.str.lower().str.contains('choi')]


# In[20]:


### Q9. world cup에 출전을 1회만 하였음에도 가장 많은 득점을 올렸던 선수는 누구인가요?


# In[32]:


goal_df2[goal_df2.NO_cup == 1].sort_values(by='Goals', ascending=False).iloc[0, 0]


# In[21]:


### 10. world cup에 출전을 1회만 한 선수가 가장 많은 국가는 어디일까요?


# In[33]:


goal_df2[goal_df2.NO_cup == 1]['Country'].value_counts().index[0]


# In[22]:


# end of files

