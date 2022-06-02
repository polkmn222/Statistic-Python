#!/usr/bin/env python
# coding: utf-8

# ## if문이란?

# > 돈이 있으면, 택시를 타고,
# > 돈이 없으면, 걸어가야 한다.

# In[1]:


money = True
if money:
    print('택시')
else:
    print('걷기')


# ### if문의 생김새 

# ![image-2.png](attachment:image-2.png)

# In[2]:


### 들여쓰기 (indentation)


# In[3]:


money = True
if money:
    print('시계')
    print('팔찌')
print('차')
    print('집')


# ## 비교연산자

# ![image.png](attachment:image.png) 

# > 만약 3800원 이상의 돈을 갖고 있다면, 택시를 타고
# > 아니면 걸어가세요

# In[4]:


money = 0
if money >= 3800:
    print('택시')
else:
    print('걷기')


# In[5]:


## - 1번 만약에 카드는?
money = 0
card = True

if money >=3800 or card:
    print('택시')

else:
    print('걷기')


# In[6]:


## - 2번 :: Bag의 개념을 적용?
bag = ['card','gold','silver','phone']

if money>=3800 or 'card' in bag:
    print('택시')

else:
    print('걷기')


# ### 다양한 조건을 표현하는 elif

# In[7]:


bag1 = ['phone','ring']

if 'money' in bag1:
    print('택시')

else: 
    if 'card' in bag1:
        print('택시')
    else:
        if 'gold' in bag1:
            print('택시')
        else:
            print('걷기')


# In[8]:


bag1 = ['phone','ring']

if 'money' in bag1:
    print('택시')
elif 'card' in bag1:
    print('택시')    
elif 'gold' in bag1:
    print('택시')
else:
    print('걷기')


# ### 조건부 표현식

# In[9]:


score = 55 # 오점이 없는 여러분


# In[10]:


if score >= 60:
    message = 'success'
else:
    message = 'fail'


# In[11]:


score = 95


# In[12]:


### 위의 식을 한줄로 표현해보자.

# step1 두 줄로 먼저 만들자
if score >= 60: message = 'success'
else: message = 'fail'

# step2 한 줄로 써보기
# if score >= 60: message = 'success' else: message = 'fail'
        
# step3 도치문장... 
# message = 'success' if score >= 60  else message = 'fail'

# step4 뒤의 변수를 지워보기... 
message = 'fail' if score < 60  else('good' if score<75 else 'very_good')

print(message)


# In[13]:


# end of file

