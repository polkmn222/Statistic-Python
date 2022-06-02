#!/usr/bin/env python
# coding: utf-8

# In[16]:


# Q1. 데이터를 로딩해보세요 ^^
# chipotle.tsv
# 확장자가 csv가 아닌점에 유의하세요

import pandas as pd
chipo_df = pd.read_csv('./chipotle.tsv', sep='\t')
chipo_df.head(2)

# csv Comma Seperate Value
# tsv Tab Seperate Value


# In[6]:


# Q2. quantity의 컬럼값이 3인 데이터를 추출해보세요.
# 그리고 그 데이터의 첫 5행만 가져와보세요


# In[20]:


chipo_df[chipo_df['quantity']==3].head(5)


# In[7]:


# Q3. Q2번에서 가져온 데이터의 index를 재정의 해보세요.
# reset_index를 활용


# In[21]:


chipo_df[chipo_df['quantity']==3].head(5).reset_index()


# In[8]:


#Q4. quantity, item_name 두 개의 컬럼으로 구성된 새로운 df를 정의해주세요 ^^


# In[22]:


answer_q4 = chipo_df[['quantity', 'item_name']]
print(answer_q4)


# In[10]:


## 난이도 상
# Q5. item_price값에서 달러표시를 제거하고, float64로 데이터를 바꾸어
# new_price라고 하는 컬럼값에 넣어서
# chipo_df를 재정의 해보세요.


# In[34]:


chipo_df['new_price'] = chipo_df.item_price.str[1:].astype('float64')
chipo_df.head(5)
chipo_df.info()


# In[12]:


# Q6. new_price 컬럼이 5이하의 값을 갖는 df를 추출하고 
# 전체 obs 개수를 구해보세요


# In[37]:


chipo_df[chipo_df.new_price <= 5].shape[0]


# In[15]:


# Q7. item_name 명이 Chicken Salad Bowl 인 df를 추출하고
# index를 재정의해보세요


# In[39]:


chipo_df[chipo_df.item_name == 'Chicken Salad Bowl'].reset_index()


# In[16]:


# Q8. new_price 값이 9이하이고 item_name값이 Chicken Salad Bowl인 df
# 를 추출해보세요.


# In[43]:


# chipo_df[chipo_df.new_price <=9 & chipo_df.item_name == 'Chicken Salad Bowl']
chipo_df[(chipo_df.item_name == 'Chicken Salad Bowl') & (chipo_df.new_price <=9)]


# In[17]:


# Q9. chipo_df의 new_price 컬럼 값을 오름차순 :: 기억하자 ascending!!!
# 으로 정렬하고 index를 재정의해보세요.


# In[44]:


chipo_df.sort_values(by = 'new_price', ascending=True)


# In[18]:


# 난이도 중상
# Q10. chipo_df의 item_name 컬럼 값 중에서 Chips를 포함하는 
# 전체 경우의 데이터를 출력해보세요.


# In[47]:


chipo_df[chipo_df.item_name.str.contains('Chips')]['item_name'].value_counts()


# In[20]:


chipo_df.head(2)


# In[ ]:


# Q11. chipo_df의 new_price 컬럼값에 따라 내림차순으로 정리하고
# index를 재정의 해보세요


# In[48]:


chipo_df.sort_values(by = 'new_price', ascending=False).reset_index()


# In[ ]:


# Q12. chipo_df의 item_name의 column 값이 Steak Salad  또는 Burrito인
# 데이터를 인덱싱해보세요.


# In[75]:


chipo_df[(chipo_df.item_name == 'Steak Salad') | (chipo_df.item_name == 'Burrito')].reset_index()
# chipo_df[(chipo_df.item_name == 'Steak Salad')]
# chipo_df[(chipo_df.item_name == 'Burrito')]


# In[22]:


# Q13. chipo_df의 new_price에서 평균값 이상의 데이터들만 인덱싱해보세요


# In[96]:


chipo_df['new_price'].mean()
chipo_df[chipo_df.new_price >= chipo_df['new_price'].mean()].reset_index()


# In[23]:


# Q14. chipo_df에서 결측치를 확인해보세요.


# In[97]:


# chipo_df[chipo_df.choice_description == 'Nan']
chipo_df.isna().sum()


# In[85]:


# Q15. 결측치가 있다면 결측치 값을 'N'으로 대체해보세요


# In[87]:


chipo_df['choice_description'] = chipo_df['choice_description'].fillna('N') # -1
chipo_df


# In[ ]:


chipo_df.choice_description.fillna('N', inplace=True) # -2
chipo_df.isna().sum()


# In[25]:


# Q16. chipo_df choice_description의 값에 Black이 들어가는 경우를
# 인덱싱해보세요.


# In[95]:


chipo_df[chipo_df.choice_description.str.contains('Black')].reset_index()


# In[26]:


# end of file

