#!/usr/bin/env python
# coding: utf-8

# ### 크롤러와 URL

# 크롤러(Crawler)는 웹페이지에 존재하는 하이퍼링크를 따라 돌아야 한다. 브라우저에서는 링크를 클릭하면 되지만 크롤러로 링크를 돌아 다니려면 URL과 관련된 기초적인 지식을 이해하고 활용해야 한다

# #### URL 기초 지식

# 크롤러로 링크를 돌아다니려면 링크를 나타내는 a 태그의 href 속성에서 다른 페이지의 URL을 추출해야 한다. 만약 URL이 상대 URL이라면, 절대 URL로 변환해야 한다.

# **URL은 Uniform Resource Locater**의 약자이다 URL 구조 정의를 RFC3986으로 살펴보자

# ![image.png](data:image/png;base64,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bGwMt2/fhqZpUFXVmFfTNNfbpLKWlpaQTCYtB1tVVV/W7WZkZATXr183tqkoCubn5/H666+7LvP+++8jEomwQ76PfvjDHwIAbt68afl+enoamUzGyGv6u5ROnDghXM/ExATW19cteSgWi2Fzc9P1eF24cMG3fh0jIyOO80NPw/r6umVkpnaMjY0hnU573g3QH7n1GjyiVCqhVqt5jkZE/eOxxx5DrVZzVEZ1R44cMd5JBnxddt+6dctSzrdSLvp5HgEsZ4eFW9kPAOfPn0cmk3GU59evXzdGPbP7p3/6J6PxF9i7ZmQyGdeR8+LxONbX1y15V6//5PP5ln/Xj3/8YwB7g0q40X+zvg8GWaOySVazdYLp6Wnk83nEYjHjO/29TH//939vpK1TOl2vPXHihOcjolevXu1KtwE/3PfnMeR7WjabxezsLMrl8sCO5tEplUoF+/fvRyaTYadQnyUSCVy9ehV37twJOikDQR9FbXt72/X2Pff54BkdHcWTTz5puTgXCgVMTk4imUw23Sk+CCxnh0sr5VAqlcLc3Bzy+XxPdsQfHx/Hzs6O5+BFonN1kPXa711eXoamaT2bh9rVb3X9nr8DBey1uKiq6ujjRI2tra1BVVVe1Dvg5MmTxsAH1J5qterar8U8Tzab5fC6A+bcuXOWPiP9iOXscBm0sr9QKKBYLHrmX31485MnT3YxZcHqtbJJ7zc1qI8Jv/3229A0rS+CJ6BPAihg77lQ/WWbJKdSqWBubs54ppb8pT/z/swzzwSdlL5WrVZx5swZ1Go1z+DozJkzrKQOIL2B7MyZM0EnpSUsZ4fPIJX9lUrFGLTHKzh65plnMD8/3zeVWz8EUTbpA4SY67r6ACF6X7pBfEw4m81ia2urrwL0vgmgpqenoWkaK09N0DsETk9PB52UgfX8889DURQsLCwEnZS+pHeATqfTiEQiruf3xsYGstksLl++3OUUUjdcvnwZ2WzWdWSmXsZydjj1e9lfKBSMAQL0AYPcgqOFhQUoioLnn3++y6kMXrfLpkqlAk3TLAOk6QOERCIRnD17tivp6KZKpYJEIoGlpaW+CtD7og8UEQ0mfTQ+TdNw9uzZgWxZo+b1Wx8ookZ6rQ+Ufo4pioLTp0/zPOsRlUoFr776Kq5evWoMfR+JRHDkyBEcO3aM18gewgCKiIiIiIhIUt88wkdERERERBQ0BlBERERERESSGEARERERERFJYgBFREREREQkiQEUERERERGRJAZQREREREREkhhAERERERERSWIARUREREREJIkBFBERERERkSQGUERERERERJIYQBEREREREUliAEVERERERCSJARQREREREZEkBlBERERERESSGEARERERERFJYgBFREREREQk6f8DGtZDr+Q93UcAAAAASUVORK5CYII=)

# |URL구성요소|설명|
# |---|---|
# |스키마| http 또는 https와 같은 프로토콜을 나타낸다|
# |어서리티| // 뒤에 나오는 일반적인 호스트 이름을 나타낸다. 사용자 이름, 비밀번호, 포트 번호 등을 포함하는 경우도 존재한다.|
# |경로| /로 시작하는 해당 호스트 내부에서의 리소스 경로를 나타낸다.|
# |쿼리|? 나오는 경로와는 다른 방법으로 리소스를 표현하는 방법이다. 존재하지 않는 경우도 있다.|
# |플래그먼트|뒤에 나오는 리소스 내부의 특정 부분 등을 나타낸다. 존재하지 않는 경우도 있다.|

# 

# ### 절대 URL과 상대 URL

# 절대 URL과 상대 URL를 정의하는 기준은 매우 다양할 수 있으나, http:// 등의 스키마로 시작하는 URL을 절대 URL이라 정의한다.
# 
# 
# 그리고 이러한 절대 URL을 기준으로 상대적인 경로를 표현하게 되는 URL을 상대 URL이라 정의한다

#     * 상대 URL1 - //로 시작하는 URL
# 
#     * 상대 URL2 - /로 시작하는 URL
# 
#     * 이 밖의 형식

# In[1]:


# 상대 URL을 절대 URL로 변환하기 :: urllib.parse 모듈 하의 urljoin()함수 활용

from urllib.parse import urljoin

base_url = 'http://example.com/books/top.html'


# In[2]:


urljoin(base_url, '//cdn.example.com/logo.png') # //로 시작하는 url


# In[3]:


urljoin(base_url, '/articles/') # /로 시작하는 URL


# In[4]:


# 파이썬으로 크롤러 만들기
# 1. Requests로 웹 페이지추출
# 2. lxml로 웹 페이지 스크레이핑
# 3. 크롤링 대상 :: 한빛 미디어의 "새로나온 책 목록"

'https://www.hanbit.co.kr/store/books/new_book_list.html' ## :: 예시를 위한 사이트


# In[5]:


get_ipython().system('pip install cssselect')


# In[6]:


import requests
import lxml.html

response = requests.get('https://www.hanbit.co.kr/store/books/new_book_list.html')
root = lxml.html.fromstring(response.content)
for a in root.cssselect('.view_box a'):
    url = a.get('href')
    print(url)


#     1) 위의 javascript:;라는 부분은 자바스크립트 코드를 수행하라는 의미인데, 이는 상세 페이지 이동과 전혀 관계가 없다. 따라서 이를 제외할 수 있으면 좋겠다.
# 
#     2) 또한 상대 url들을 절대 URL로 변환해야 한다. urljoin() 외에 lxml.html.HtmlElemen 클래스를 활용할 수도 있다. make_links_absolute()메서드를 활용하자.

# In[7]:


import requests
import lxml.html

response = requests.get('https://www.hanbit.co.kr/store/books/new_book_list.html')
root = lxml.html.fromstring(response.content)

# 모든 링크를 절대 URL로 변환하자.
root.make_links_absolute(response.url)

for a in root.cssselect('.view_box .book_tit a'):
    url = a.get('href')
    print(url)


# ### **세션이란?**
# 
# 웹 브라우저와 서버가 HTTP 프로토콜을 통해서 하는 모든 커뮤니 케이션은 무상태(Stateless)라고 한다. Client와 Server 사이의 메세지가 완벽하게 독립적이라는 의미이다. 

# 세션이라는 것은 사이트와 특정 브라우저 사이의 "state"를 유지시키는 것이다. 세션이 사용자가 매 브라우저마다 임의의 데이터를 저장하게 하고, 이 데이터가 브라우저가 접속할 때마다 사이트에서 활용할 수 있도록 한다.

# In[8]:


## 위의 내용들 Crawler의 사용자 정의 함수로 정의하여 생성

import requests
import lxml.html

def main():
    """
    크롤러의 메인 처리
    """

    # 여러 페이지에서 크롤링할 것이므로 Session을 사용해보자.
    session = requests.Session()
    
    # scrape_list_page()함수를 호출해서 제너레이터를 추출
    response = session.get('https://www.hanbit.co.kr/store/books/new_book_list.html')
    urls = scrape_list_page(response)
    
    # 제너레이터는 list처럼 사용할 수 있다.
    for url in urls:
        print(url)

def scrape_list_page(response):
    root = lxml.html.fromstring(response.content)
    root.make_links_absolute(response.url)
    for a in  root.cssselect('.view_box .book_tit a'):
        url = a.get('href')
        #yield 구문으로 제너레이터의 요소 반환
        yield url ##  return으로 결과값을 받아줄 수 있으나
                  ##  yield로도 결과값을 받아 줄 수 있습니다.


if __name__ == '__main__' :
    main()


# In[9]:


session = requests.Session()

response = session.get('https://www.hanbit.co.kr/store/books/new_book_list.html')
response

urls = scrape_list_page(response)
for url in urls:
    print(url)


# ### **상세 페이지에서 스크레이핑**

# In[10]:


import requests
import lxml.html
## 위의 내용들 Crawler의 사용자 정의 함수로 정의하여 생성

import requests
import lxml.html

def main():
    """
    크롤러의 메인 처리
    """

    # 여러 페이지에서 크롤링할 것이므로 Session을 사용해보자.
    session = requests.Session()
    
    # scrape_list_page()함수를 호출해서 제너레이터를 추출
    response = session.get('https://www.hanbit.co.kr/store/books/new_book_list.html')
    urls = scrape_list_page(response)
    
    # 제너레이터는 list처럼 사용할 수 있다.
    for url in urls:
        print(url)

def scrape_list_page(response):
    root = lxml.html.fromstring(response.content)
    root.make_links_absolute(response.url)
    for a in  root.cssselect('.view_box .book_tit a'):
        url = a.get('href')
        #yield 구문으로 제너레이터의 요소 반환
        yield url ##  return으로 결과값을 받아줄 수 있으나
                  ##  yield로도 결과값을 받아 줄 수 있습니다.

def scrape_detail_page(response):
    
    '''
    상세 페이지의 Response에서 책 정보를 dict로 추출한다
    '''

    root = lxml.html.fromstring(response.content)
    ebook = {
        'url':response.url,
        'title':root.cssselect('.store_product_info_box h3')[0].text_content(),
        'price':root.cssselect('.pbr strong')[0].text_content(),
        'content':[p.text_content()\
                   for p in root.cssselect('#tabs_3.hanbit_edit_view p')]
    }
    return ebook


if __name__ == '__main__' :
    main()


# In[11]:


# end of file - 한빛출판사


# In[11]:




