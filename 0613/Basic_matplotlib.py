#!/usr/bin/env python
# coding: utf-8

# # 01. Matplotlib 기본 사용

# In[1]:


# 맷플롯립 라이브러리 호출-1

import matplotlib.pyplot as plt

plt.plot([1,2,3,4])
plt.show()

## 예제 -2 
# X축 = 1,2,3,4
# y축 = 1,4,9,16

plt.plot([1,2,3,4],[1,4,9,16])
plt.show()


# ### 스타일 지정법

# plt.plot([1,2,3,4],[1,4,9,16],'ro')  red의 원형 마커(marker)
# 
# 
# plt.axis([xmin, xmax, ymin, ymax])

# In[2]:


plt.plot([1,2,3,4],[1,4,9,16],'ro')

plt.axis([0, 6, 0, 20])


# ## 하나의 면에 여러 그래프 그리기

# plt.plot(X1,y1,'r--', X2, y2, 'bs', X3, y3, 'g^')  red의 점선을 추가 
#       
#                                                    bs는 파란 사각형
#                                                    
#                                                    g^는 녹색 삼각형

# In[3]:


import matplotlib.pyplot as plt
import numpy as np

# 200ms 간격으로 균일된 시간이 있다고 가정
t = np.arange(0,5, 0.2) # 0~5까지의 데이터가 0.2의 간격

# 위에서 예를 든 것처럼 빨간 대쉬, 파란 사각형, 녹색의 삼각형
plt.plot(t,t,'r--', t,t**2, 'bs', t,t**3, 'g^')
plt.show()


# ## 컬럼이나 key값이 존재하는 데이터 활용

# In[4]:


import matplotlib.pyplot as plt

data_dict = {'X_train':[1,2,3,4,5],
            'y_train':[2,3,5,10,8]}

plt.plot('X_train', 'y_train', data=data_dict)
plt.show()


# # 02. 축 레이블 설정

# ![image-2.png](attachment:image-2.png)

# In[5]:


import matplotlib.pyplot as plt

plt.plot([1,2,3,4],[1,4,9,16])
plt.xlabel('X-label', labelpad=15, fontdict={'family':'serif',
                                            'color':'b',
                                            'weight':'bold',
                                            'size':25}) #labelpad는 여백지정
plt.ylabel('y-label', labelpad=25) #labelpad는 여백지정
plt.show()


# In[6]:


### X축, y축의 label 위치지정하기


# In[7]:


import matplotlib.pyplot as plt

plt.plot([1,2,3,4],[1,4,9,16])

### loc의 기본 3가지 위치지정
## Xlabel의 경우 :: ['left','center','right']
## ylabel의 경우 :: ['top','center','bottom']

plt.xlabel('X-label', labelpad=15, loc='right') #labelpad는 여백지정
plt.ylabel('y-label', labelpad=25, loc='top') #labelpad는 여백지정
plt.show()


# In[8]:


import matplotlib.pyplot as plt

plt.plot([1,2,3,4],[2,3,5,10], label = 'price($)')
plt.xlabel('X-axis', labelpad= 15)
plt.ylabel('y-axis', labelpad= 20)
plt.legend()
plt.show()


# ### 위치 지정하기

#  ![image.png](attachment:image.png) 

# In[9]:


## 만약 여러분께서 다른 함수나 API에 대한 설명이 필요하실때
## help 함수를 적용해주십시오
help(plt.legend)


# In[10]:


# 범례의 위치 지정시 - 1

import matplotlib.pyplot as plt

plt.plot([1,2,3,4],[2,3,5,10], label = 'price($)')
plt.xlabel('X-axis', labelpad= 15)
plt.ylabel('y-axis', labelpad= 20)

# plt.legend(loc=(1.0,1.0)) # 데이터 영역에서 왼쪽 아래에 해당
# plt.legend(loc=(1.0,1.0)) #데이터 영역에서 오른쪽 위의 위치에 해당
plt.show()


# In[11]:


# 범례의 위치 지정시 - 2

import matplotlib.pyplot as plt

plt.plot([1,2,3,4],[2,3,5,10], label = 'price($)')
plt.xlabel('X-axis', labelpad= 15)
plt.ylabel('y-axis', labelpad= 20)

plt.legend(loc='best') # 데이터 영역에서 왼쪽 아래에 해당

plt.show()


# ## column 개수 지정

# ![image.png](attachment:image.png) 

# In[12]:


import matplotlib.pyplot as plt

plt.plot([1,2,3,4],[2,3,5,10], label = 'price($)')
plt.plot([1,2,3,4],[3,5,9,7], label = 'demand($)')
plt.xlabel('X-axis', labelpad= 15)
plt.ylabel('y-axis', labelpad= 20)
plt.legend(loc='best', ncol= 2, fontsize=12,frameon=True, shadow=True) # fontsize로 해당 
                                             # 범례의 크기조절가능
                                             # 범례 테두리 꾸미기
                                             # frameon=True, shadow=True
plt.show()


# # Matplotlib 축 범위 지정 

# ![image.png](attachment:image.png)

# * xlim() :: x축이 표시되는 범위를 지정하거나 반환한다.
# 
# * ylim() :: y축이 표시되는 범위를 지정하거나 반환한다.
# 
# * axis() :: x축, y축이 표시되는 범위를 지정하거나 반환한다.

# ![image.png](attachment:image.png)

# In[13]:


import matplotlib.pyplot as plt

plt.plot([1,2,3,4], [2,3,5,10])
plt.xlabel('x-axis')
plt.ylabel('y-axis')
# plt.xlim([0,5]) # 축의 범위 ([xmin, Xmax])
# plt.ylim((0,20)) # 축의 범위 ((ymin, ymax)]
plt.axis([0,5,0,20])

plt.show()


# # Matplotlib 선 종류 지정

# ![image.png](attachment:image.png) 

# In[14]:


# 예제1

import matplotlib.pyplot as plt

plt.plot([1,2,3],[4,4,4], '-', color= 'C0', label='solid')
plt.plot([1,2,3],[3,3,3], '--', color= 'C0', label='Dashed' )
plt.plot([1,2,3],[2,2,2], ':', color= 'C0', label='Dotted')
plt.plot([1,2,3],[1,1,1], '-.', color= 'C0', label='Dash-dot')


# In[15]:


# 예제2

import matplotlib.pyplot as plt

plt.plot([1,2,3],[4,4,4], linestyle = 'solid', color= 'C0', label='solid')
plt.plot([1,2,3],[3,3,3], linestyle = 'dashed', color= 'C0', label='Dashed' )
plt.plot([1,2,3],[2,2,2], linestyle = 'dotted', color= 'C0', label='Dotted')
plt.plot([1,2,3],[1,1,1], linestyle = 'dashdot', color= 'C0', label='Dash-dot')


# Matplotlib Tutorial - 파이썬으로 데이터 시각화하기

# https://wikidocs.net/book/5011 

# Matplotlib 공식 사이트 

# https://matplotlib.org/

# 토닥토닥 파이썬

# https://wikidocs.net/book/2454

# In[16]:


# end of file

