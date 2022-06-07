#!/usr/bin/env python
# coding: utf-8

# In[3]:


# Q1. Chipolet.csv 데이터 로딩을 해주세요.
import pandas as pd
chipo_df = pd.read_csv('./chipotle.tsv', sep = '\t')


# In[3]:


# Q2. df의 item_name 컬럼 값이 Steak Salad 또는 Bowl 인 데이터를 데이터 프레임화 한 후, 
# item_name를 기준으로 중복행이 있으면 제거하되 첫번째 케이스만 남겨주세요

# drop_duplicates를 한 번 검색해서 써보셔요 ^^


# In[11]:


Answer1 = chipo_df[(chipo_df.item_name=='Steak Salad')|(chipo_df.item_name=='Bowl')]
Answer1.drop_duplicates('item_name')


# In[4]:


# Q3. df의 item_name 컬럼 값이 Steak Salad 또는 Bowl 인 데이터를 데이터 프레임화 한 후, 
# item_name를 기준으로 중복행이 있으면 제거하되 마지막 케이스만 남겨주세요

# drop_duplicates를 한 번 검색해서 써보셔요 ^^


# In[13]:


Answer1 = chipo_df[(chipo_df.item_name=='Steak Salad')|(chipo_df.item_name=='Bowl')]
Answer1.drop_duplicates('item_name', keep='last')


# In[5]:


# Q4. df의 데이터 중 item_name값이 Izze데이터를 Fizzy Lizzy로 수정해보세요.


# In[18]:


# 1번은 기초 인덱싱 
chipo_df[chipo_df.item_name=='Izze']['item_name'] = 'Fizzy Lizzy'


# In[19]:


# 2번은 loc
chipo_df.loc[chipo_df.item_name=='Izze','item_name'] = 'Fizzy Lizzy'


# In[20]:


# 3번은 apply lambda
chipo_df['item_name'].apply(lambda x: 'Fizzy Lizzy' if x=='Izze' else x)


# In[6]:


# Q5. df의 데이터 중 item_name 값의 단어갯수가 15개 이상인 데이터를 인덱싱해보세요.


# In[29]:


chipo_df[chipo_df['item_name'].apply(lambda x : len(x))>=15]


# In[24]:


chipo_df[chipo_df.item_name.str.len()>=15]


# ### Airbnb 데이터를 로드하고 상위 5개 컬럼을 출력해보세요

# In[8]:


### Groupby 함수를 활용한 문제입니다.
### 난이도가 조금 있어요~ ^^


# In[31]:


import numpy as np
import pandas as pd
air_df = pd.read_csv('./air_bnb_data/AB_NYC_2019.csv')


# In[7]:


# Q1  데이터의 각 host_name의 빈도수를 구하고 host_name으로 정렬하여 상위 5개를 출력해보세요.


# In[41]:


## Answer_1
air_df.groupby(by='host_name')['name'].count().head()


# In[35]:


## Answer_2
air_df.groupby(by='host_name').size().head()


# In[38]:


## Answer_3
air_df['host_name'].value_counts().sort_index().head()


# In[9]:


# Q2  neighbourhood_group의 값에 따른 neighbourhood컬럼 값의 갯수를 구해보세요


# In[48]:


## Answer_1 
# 참고 as_index 파라미터를 조절해보세요.
# 참고2 count 및 size를 비교해보세요.

air_df.groupby(by=['neighbourhood_group','neighbourhood'],as_index=False)['id'].count()


# In[53]:


# Q3  neighbourhood_group의 값에 따른 price값의 평균, 분산, 최대, 최소값을 구해보세요.
# Answer1
air_df.groupby(by=['neighbourhood_group'])['price'].agg(['mean','var','max','min'])


# In[54]:


# Answer2
air_df[['neighbourhood_group','price']].groupby(by=['neighbourhood_group']).agg(['mean','var','max','min'])


# ![image.png](attachment:image.png) 

# In[12]:


# Q4  neighbourhood_group의 값과 neighbourhood에 따른 price의 평균을 구해보세요.


# In[68]:


air_df.groupby(['neighbourhood_group','neighbourhood'])['price'].mean()


# ### 카드 사용 데이터 

# In[13]:


### Apply, Lambda 문제


# In[14]:


# Q1 데이터를 로딩해주세요.


# In[73]:


credit_df = pd.read_csv('.//credit_card/BankChurners.csv')


# In[76]:


credit_df['Income_Category'].value_counts()


# In[ ]:


# Q2 Income_Category의 카테고리를 map 함수를 이용하여 다음과 같이 변경하여 newIncome 컬럼에 매핑해주세요.
# # Unknown : N 
# Less than 40K:a
#     40K - 60K:b
#         60K - 80K:c
#             80K - 120K:d
#                 120K +’ : e


# ![image.png](attachment:image.png) 

# In[85]:


dict1 = {'Unknown':'N',
        'Less than $40K':'A',
        '$40K - $60K':'B',
        '$60K - $80K':'C',
        '$80K - $120K':'D',
        '$120K +':'E'}

credit_df['New_income'] = credit_df['Income_Category'].map(lambda x : dict1[x])
credit_df.head()


# In[15]:


# Q3 Customer_Age의 값을 이용하여 나이 구간을 AgeState 컬럼으로 정의해보세요. 
# (0~9 : 0 , 10~19 :10 , 20~29 :20 … 각 구간의 빈도수를 출력하라


# In[99]:


credit_df['Age_freq'] = credit_df.Customer_Age.map(lambda x : (x//10)*10)
credit_df.head()

credit_df.Age_freq.value_counts().sort_index()


# In[19]:


# Q4 Education_Level의 값중 Graduate단어가 포함되는 값은 1 그렇지 않은 경우에는 0으로 변경하여
# newEduLevel 컬럼을 정의하고 빈도수를 출력해주세요.

# if문 한줄로 쓰셔서...
# apply_lambda 및 value_counts() 활용 ^^


# In[100]:


credit_df.Education_Level.apply(lambda x : 1 if 'Graduate' in x else 0)


# In[20]:


# Q5 Credit_Limit 컬럼값이 4500 이상인 경우 1 그외의 경우에는 모두 0으로 하는 newLimit 정의하시고
# newLimit 각 값들의 빈도수를 출력해주세요 ^^

# if문 한줄로 쓰셔서...
# apply_lambda 및 value_counts() 활용 ^^


# In[101]:


credit_df.Credit_Limit.apply(lambda x : 1 if x>=4500 else 0)


# In[21]:


# Q6 Gender 컬럼값 M인 경우 male F인 경우 female로 값을 변경하여 Gender 컬럼에 새롭게 정의해주세요.
# 각 value의 빈도를 출력해보세요 ^^

# if문 한줄로 쓰셔서...
# apply_lambda 및 value_counts() 활용 ^^


# In[102]:


credit_df.Gender.apply(lambda x: 'male' if x=='M' else 'female')


# In[22]:


# end of file

