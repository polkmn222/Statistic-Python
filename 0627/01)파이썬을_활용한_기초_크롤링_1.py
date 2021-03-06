# -*- coding: utf-8 -*-
"""01)파이썬을 활용한 기초 크롤링-1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/19KSu0P3mBKQH-O6kFeRQibYK2p72zrIr

## 1.1 urllib으로 웹 페이지 추출하기

urllib.request를 사용하면 웹 페이지를 간단하게 추출할 수 있습니다. 하지만 HTTP 헤더를 마음대로 변경하려 하거나 Basic인증을 사용하려면 복잡한 처리를 해야 합니다.
"""

from urllib.request import urlopen
f = urlopen('http://hanbit.co.kr')
# urlopen()함수는 HTTPResponse 자료형의 객체를 반환한다.
# 이 객체는 파일 객체이므로 open( )함수를 반환되는 파일 객체처럼 다루면 된다.
type(f)

# read()메서드로 HTTP 응답 본문(bytes 자료형)을 추출한다.
# HTTP 연결은 자동으로 닫히기에 따로 python의 close()함수를 호출하지 않아도 된다.

f.read()

f.status
# 200이라는 상태코드는 제대로 연결이 되었음을 의미한다.

f.getheader('Content-Type')

"""## 1.2 문자 코드 다루기

HTTPresponse.read() 메서드로 추출할 수 있는 응답 본문의 값은 bytes 자료형이므로 문자열(str형태입니다)로 다루려면 문자 코드를 지정해서 디코딩해야 합니다.

최근에는 HTML5의 기본 인코딩 방식인 UTF-8로 작성된 웹페이지가 많으므로 UTF-8을 전제로 디코딩하는 것도 좋습니다.

다만, 한국어를 포함하는 사이트를 크롤링할 시에는 여러 개의 인코딩이 섞여 있을 수도 있으므로 HTTP헤더를 참조해서 적절한 인코딩 방식으로 디코딩 해야 합니다.

## HTTP 헤더에서 인코딩 방식 추출하기

HTTP 응답의 Content-Type 헤더를 참조하면 해당 페이지에 사용되고 있는 인코딩 방식을 찾아낼 수 있습니다. 한국어가 포함된 페이지는 일반적으로 다음과 같은 Content-Type 헤더를 가지고 있습니다.

* text/html

    * text/html;charset=UTF-8

    * text/html;charset=EUC-KR

charset= 뒤에 적혀 있는 UTF-8과 EUC-KR이 해당 페이지의 인코딩 방식입니다. 인코딩이 명시되어 있지 않은 경우에는 UTF-8로 다루면 됩니다. 

만약 Content-Type 헤더의 값에서 인코딩을 추출할 때는 정규표현식을 사용할 수 있습니다. 

하지만 HTTPResponse.info() 메서드로 추출할 수 있는 HTTPMessage 객체의 get_content_charset() 메서드를 사용하면 더 쉽게 추출이 가능합니다. 

이를 아래의 코드로 활용해보겠습니다.
"""

import sys
from urllib.request import urlopen
f = urlopen('https://www.hanbit.co.kr/store/books/full_book_list.html')

# HTTP 헤더를 기반으로 인코딩 방식을 추출합니다(명시돼 있지 않을 경우 utf-8을 사용하게 합니다.)
encoding = f.info().get_content_charset(failobj='utf-8')

# 인코딩 방식을 표준 오류에 출력합니다.
print('encoding:' , encoding, file=sys.stderr)

# 추출한 인코딩 방식으로 디코딩합니다.
text = f.read().decode(encoding)

# 웹 페이지의 내용을 표준 출력에 출력합니다.
print(text)

## 파일을 urlopen_encoding.py라는 이름으로 저장하고 실행하면
## HTTP 헤더에서 추출된 인코딩 방식과 디코딩된 문자열이 출력됨

## 저장된 위치로 cd(Change directory)를 활용하여 이동
## python urlopen_encoding.py> dp.html

"""### meta 태그에서 인코딩 방식 추출하기

HTTP 헤더에서 추출하는 인코딩 정보가 항상 올바른 것은 아닙니다. 웹 서버 설정을 제대로 하지 않았다면 Content-Type 헤더의 값과 실제 사용되고 있는 인코딩 형식이 다를 수 있습니다.

일반적인 브라우저는 HTML 내부의 meta 태그 또는 응답 본문의 바이트열도 확인해서 최종적인 인코딩 방식을 결정하고 화면에 출력합니다. 디코딩 처리때 UnicodeDecodeError가 발생한다면 이러한 처리를 모방해서 구현하면 해결이 가능합니다.

* <meta charset = "utf-8">

    * <meta http-equiv = "Content-Type" content="textm/html; charset=EUC_KR">
"""

## 아래의 코드는 meta 태그의 charset 값에서 인코딩 방식을 추출하고 디코딩하는 코드입니다.
## 정규표현식 처리를 위해 re 모듈을 활용해봅니다.

import re
import sys
from urllib.request import urlopen

f= urlopen('https://www.hanbit.co.kr/store/books/full_book_list.html')
# bytes 자료형의 응답 본문을 일단 변수에 저장한다.
bytes_content = f.read()
bytes_content[:700]

# charset은 HTML의 앞부분에 적혀 있는 경우가 많으므로
# 응답 본문의 앞부분 1024바이트를 ASCII 문자로 디코딩 해둔다.
# ASCII 범위 이외의 문자들은 U+FFFD(REPLACEMENT CHARACTER)로 변환되어 예외가 발생하지 않도록 한다
scanned_text = bytes_content[:1024].decode('ascii', errors='replace')
scanned_text 

## 디코딩한 값에서 정규표현식(re로 임포트한 내용) charset값을 추출합니다
match = re.search(r'charset=["\']?([\w-]+)', scanned_text)

if match:
    encoding = match.group(1)

else:
    # charset 값을 알 수 없거나 명시되어있지 않다면 UTF-8을 활용합니다.
    encoding = 'utf-8'

# 추출한 인코딩을 표준 오류에 출력한다
print('encoding:', encoding, file=sys.stderr)

# 추출한 인코딩으로 다시 디코딩을 수행한다.
text=bytes_content.decode(encoding)

# 응답 본문을 print 한다
print(text)

# end of file