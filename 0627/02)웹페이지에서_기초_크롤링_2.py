#!/usr/bin/env python
# coding: utf-8

# 다운로드한 웹 페이지 또는 RSS에서의 스크레이핑을 시도해보자. 파이썬 표준 라이브러리로 저장한 파일에서 도서의 제목 및 URL 등의 데이터를 추출한다.

#     * 정규표현식
# 
#     * XML 파서

# * 정규 표현식으로 Scraping할 때에는 HTML을 단순한 문자열로 취급하고, 필요한 부분을 추출한다. 제대로 마크업되지 않은 웹 페이지도 문자열의 특징을 파악하면 쉬운 스크레이핑 작업이 가능하다. 

# * XML 파서로 하는 Scraping할 때는 HTML태그를 분석(파싱:: parsing)하고, 필요한 부분을 추출한다. 블로그 또는 뉴스 사이트 정보를 전달하는 RSS처럼 많은 정보가 XML에는 제공된다.

# 참고) XML과 HTML은 비슷해보이나, XML파서에 HTML을 곧바로 넣어 분석할 수는 없다. HTML은 종료 태그(tag)가 생략되는 등 XML에 비하면 더욱 유연하게 사용되기 때문이다. 
# 
# 뿐 만 아니라, 웹 브라우저는 문법에 문제가 있는 HTML이라도 어떻게든 출력해주는 경향이 있다.
# 
# 하지만 parsing의 경우 문제가 있는 HTML은 제대로 파싱할 수 없으며, 이러한 문제가 있는 웹 페이지는 생각보다 많다.

# ***::: 따라서, HTML을 parsing할 때에는 HTML 전용 parser가 필요하다***

# In[1]:


### 정규표현식 scraping
import re

# re.search() 함수를 사용하면 두 번째 매개변수의 문자열이 첫 번째 매개변수의 정규식에 일치하는지 확인 가능하다
# 맞을 경우에는 match 객체를 반환, 그렇지 않다면, None을 반환한다.
# 아래의 예에서는 match 객체가 반환되었다.
# match='abc'를 보면 abc가 매치된 것이 확인 가능하다.
re.search(r'a.*c', 'abc123DEF')


# In[2]:


# 다음 예제는 정규표현식에 일치하지 않으므로 None을 반환한다
# interactive shell에서 결과가 없을 경우, 바로 >>>을 출력한다.
re.search(r'a.*d', 'abc123DEF')


# In[3]:


# 세 번째 매개변수로 옵션을 지정한다
# re.IGNORECASE(또는 re.I)를 지정하면 대소문자를 무시한다
# 이외에도 굉장히 다양한 옵션이 존재한다

re.search(r'a.*d', 'abc123DEF', re.IGNORECASE)


# In[4]:


# match 객체의 group()메서드로 일치한 값을 추출
# 매개변수에 0을 지정하면 매치된 모든 값을 반환
m = re.search(r'a(.*)c', 'abc123DEF')
m.group(0)


# In[5]:


# 매개변수에 1이상의 숫자를 지정하면 정규 표현식에서 ()로 감싼 부분에 해당하는 값을 추출
# 1이라면 1번째 그룹, 2라면 2번째 그룹이 추출된다.

m.group(1)


# In[6]:


# re.findall() 함수를 사용하면 정규 표현식에 맞는 모든 부분을 추출 가능하다
# 다음 예에서는 2글자 이상의 단어를 모두 추출해보자.
# \w는 유니코드로 글자를 비교한다. 이 밖에도 공백 문자는 \s 등으로 추출 가능하다
re.findall(r'\w{2,}', 'This is a pen')


# In[7]:


# re.sub()함수를 사용하면 정규 표현식에 맞는 부분을 바꿀 수 있다.
# 3번째 매개변수에 넣은 문자열에서 첫 번재 정규 표현식에 맞는 부분을
# 2번째 매개변수로 변경한다.
re.sub(r'\w{2,}','yes_good', 'This is a pen')


# In[8]:


import re
from html import unescape

# 이전 절에서 다운로드한 파일을 열고 html이라는 변수에 저장해보자.
with open('dp.html') as f:
    html = f.read()
    
# re.findall()을 사용해 도서 각각에 해당하는 HTML을 추출한다.\
for partial_html in re.findall(r'<td class="left"><a.*?<\td>', html, re.DOTALL):
    # 도서의 URL을 추출하자.
    url = re.search(r'<a href="(.*?)">', partial_html).group(1)
    print(url)

    


# In[ ]:


for partial_html in re.findall(r'<td class="left"><a.*?</td>', html, re.DOTALL):
    # 도서의 URL을 추출하자.
    
    print(partial_html)
    url = re.search(r'<a href="(.*?)">', partial_html).group(1)
    url = 'https://hanbit.co.kr/' + url 
#     print(url)
    
    # 태그를 제거하여 도서의 제목을 추출한다.
    title = re.sub(r'<.*?>','', partial_html)
    title = unescape(title)
    print('url:', url)    
    print('title:', title)
    print('---'*35)


# In[ ]:


# end of the file

