#!/usr/bin/env python
# coding: utf-8

# # 문자열 생성방법-1

# In[1]:


"Life is too short, you need python"


# In[2]:


# 1. 큰따옴표
"Hello world"


# In[3]:


# 2. 작은 따옴표
'python is interesting'


# In[4]:


# 3. 큰따옴표 3개를 넣는 방법
"""Life is too short, you need python"""


# In[5]:


# 4. 작은따옴표 3개를 넣는 방법
'''Life is too short, you need python'''


# ## Mulit-line 을 만들고자 할 때

# In[7]:


# 1. \n (Escape 코드) 이스케이프 코드
multi = "Life is too short,\nyou need python"
print(multi)


# In[10]:


# 2. 작은 따옴표 3개

multi2 = '''Life is to short,
you need python.
'''
print(multi2)

# 3. 큰 따옴표 3개
multi3 = """Life is to short,
you need python.
"""
print(multi3)


# # concatenation :: 덧셈

# In[11]:


head = 'you need'
tail = ' python'
head + tail


# In[12]:


a = [1, 2, 3]
b = [4, 5, 6]
a + b


# In[14]:


import numpy as np  # 별칭(alias)

a_1 = np.array(a)
b_1 = np.array(b)

print(a_1 + b_1) # array(배열)
c = [a + b for a, b in zip(a, b)]
print(c)


# In[16]:


## 문자열의 곱하기
b = 'python '
answer = b * 2
print(answer)


# In[17]:


# 문자열 곱하기
print('-' * 30)
print('test')
print('-' * 30)


# In[18]:


# 문자열의 길이구하기
x = 'Life is too short, you need python'
len(x)


# ## 문자열 인덱싱과 슬라이싱

# In[21]:


print(x)
x[-1]


# In[22]:


x[-0]


# In[23]:


# 슬라이싱으로 문자열 나누기
date = '20220525sunny'
year = date[0:4]
print(year)
day = date[4:8]
print(day)
weather = date[8:]
print(weather)


# In[25]:


# 참고사항
a = 'pithon'
# a[1] = 'y'
print(a[1])
result = a[:1]+'y'+a[2:]
print(result)


# # 문자열 포메팅

# In[26]:


"현재온도는 26도 입니다."
"현재온도는 16도 입니다"


# ### % 연산자를 통한 숫자 바로 대입

# In[28]:


# 숫자를 바로 대입하는 방법
"현재온도는 %d도 입니다." % 26  # decimal 십진수


# In[29]:


# 숫자를 바로 대입하는 방법
"현재온도는 %s도 입니다." % 26  # s는 str


# In[30]:


# 숫자 값을 변수로 대입하여서 적용
num = 3
"I eat %d pythons" % num


# In[31]:


# 2개 이상의 값을 넣기
num = 10
day = 'five'
"I ate %d pythons so I was sick for %s days" % (num, day)


# ## 문자열 포멧 코드

# In[33]:


## 참고사항 %%
"Accuracy is %d%%" % 95


# # 포멧코드와 숫자를 통한 정렬

# In[34]:


# 1. 정렬과 공백
print("%10s" % 'bigdata')  # 오른쪽정렬
print("%-10s" % 'bigdata') # 왼쪽정렬


# In[38]:


# 2. 소수점 표현하기
print('%f' % 3.14159212345678)
print('%0.10f' % 3.14159212345678)
print('%.2f' % 3.14159212345678)
print('%10.2f' % 3.14159212345678)


# ## format 함수를 활용한 포매팅

# In[39]:


# 숫자 바로 대입하기
"I eat {0} pythons".format(3) 


# In[40]:


# 문자열 바로 대입하기
"I eat {0} pythons".format('five') 


# In[41]:


# 숫자 값을 가진 변수로 대입하기
num1 = 3
"I eat {0} pythons".format(num1)


# In[42]:


# 숫자 값을 가진 변수로 대입하기
num1 = 3
day1 = 'six'
"I ate {0} pythons so I was sick for {1} days".format(num1, day1)


# In[43]:


import numpy as np
array1 = np.arange(1, 11)
print(array1)
array1.reshape(2, -1)


# In[46]:


# format 함수를 통한 정렬
# 왼쪽 정렬
print('{0:<10}'.format('hi'))
# 오른쪽 정렬
print('{0:>10}'.format('hi'))
# 가운데 정렬
print('{0:^10}'.format('hi'))


# In[47]:


# 왼쪽정렬(공백채우기)
print('{0:?<15}'.format('님 나 아심'))
# 오른쪽정렬(공백채우기)
print('{0:?>10}'.format('hi'))


# In[48]:


## 포멧 함수를 통한 소수점 표현
y = 3.141592123456789

print('{0:0.4f}'.format(y))  # 원칙 {0:0.4f}
print('{0:.4f}'.format(y))  # 실무 {0:.4f}

# 오른쪽 정렬을 통한 formatting
print('{0:10.4f}'.format(y))


# ## 문자열 포매팅
# 파이썬 3.6버전부터는 f문자열 포매팅 기능을 사용할 수 있다.
# (3.6미만 버전에서는 사용할 수 없는 기능이다.)

# In[50]:


name = 'psy'
age = 31
# format 함수를 통해 활용
print('제 이름은 {name}입니다. 나이는 {age}입니다.'.format(name='psy', age=31))

# f포매팅 함수를 통해 활용
print(f'제 이름은 {name}입니다. 나이는 {age}입니다.')


# In[51]:


# f 포매팅 변칙
age = 29
print(f'나는 내년이면 {age+1}살이 됩니다.')


# In[55]:


# f 포매팅을 dict로 표현해보기
dict1 = {'name':'psy', 'age':31}
print(f'제 이름은 {dict1["name"]}입니다. 나이는 {dict1["age"]}입니다.')


# In[59]:


# 정렬의 방식
print(f'{"hi":!<10}')  # 왼쪽 정렬
print(f'{"hi":!>10}')  # 오른쪽 정렬
print(f'{"hi":!^10}')  # 가운데 정렬


# ### 문자열 관련함수 우선순위1

# In[60]:


### Split 함수
a = 'Life is too short'  # 한 덩어리의 str
print(a.split())


# In[61]:


### Split 함수 -2
b = 'a!b!c!d'
print(b.split('!'))


# In[64]:


# replace 함수
a = 'Life is too short'
a.replace('Life', 'My height')
print(a)
print(a.replace('Life', 'My height'))


# In[66]:


# join 함수
test = 'abcd'
print('.'.join(test))


# ### 문자열 관련함수 우선순위2

# In[67]:


# 문자 개수 세기(count)
a = 'happy'
print(a.count('p'))


# In[68]:


# 위치 알려주기1(find)
a = 'Big Data is the best choice'
print(a.find('b'))
print(a.find('k'))


# In[70]:


# 위치 알려주기2(index)
a = 'Big Data is the best choice'
print(a.index('b'))
# print(a.index('k')) error


# ## 문자열 관련함수 우선순위3

# In[72]:


a = 'Hello'
# 소문자 -> 대문자
print(a.upper())
# 대문자 -> 소문자
print(a.lower())


# In[73]:


# end of file

