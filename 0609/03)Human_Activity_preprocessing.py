#!/usr/bin/env python
# coding: utf-8

# ## 컬럼명 전처리

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


og_name_df = pd.read_csv('./HAPT Data Set/features.txt',header=None,
                            sep='\s+',
                            names=['column_name']).reset_index()


# In[3]:


# step1 원본 데이터를 컬럼명으로 groupby했음
# step2 cumcount라는 함수로 중복된 값을 확인 ---> 결과값 series
cum_name_sr = og_name_df.groupby(by='column_name').cumcount()

# step3 위의 결과를 보기 편하게 df로 세팅
new_name_df = pd.DataFrame(cum_name_sr, columns=['copy_cnt'])
new_name_df = new_name_df.reset_index()

name_copy_cnt = pd.merge(og_name_df,new_name_df, how='outer')


# In[4]:


### apply lambda를 적용하여 새로운 컬럼명 생성

## 중복값이 얼마나 되는지 확인한 조건식
name_copy_cnt[name_copy_cnt.copy_cnt>0]

# apply_lambda 적용
# 귀찮으시면 여기까지 -1
name_copy_cnt[['column_name','copy_cnt']].apply(lambda x: str(x[0])+'_'+str(x[1]), axis=1)

# 조금 더 전처리를 원하시면? - 2
name_copy_cnt['column_name'] = name_copy_cnt[['column_name','copy_cnt']].apply(lambda x: str(x[0])+'_'+str(x[1]) if int(x[1]) >0 else x[0], axis=1)
name_copy_cnt[name_copy_cnt['copy_cnt']>0]


# In[5]:


ftr_name = name_copy_cnt.column_name.values.tolist()


# In[7]:


### 이제부터 X_train과 X_test, y_train, y_test 데이터를 토대로...
## 분석을 수행해봅니다.

X_train = pd.read_csv('./HAPT Data Set/train/X_train.txt',header=None,
                            sep='\s+',
                            names=ftr_name)

X_test = pd.read_csv('./HAPT Data Set/test/X_test.txt',header=None,
                            sep='\s+',
                            names=ftr_name)

y_train =  pd.read_csv('./HAPT Data Set/train/y_train.txt',header=None,
                            sep='\s+',
                            names=['action'])

y_test =  pd.read_csv('./HAPT Data Set/test/y_test.txt',header=None,
                            sep='\s+',
                            names=['action'])


# In[8]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings(action='ignore')


# In[14]:


# 필요 라이브러리
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

### 객체화
dt_clf = DecisionTreeClassifier(random_state=11)
rf_clf = RandomForestClassifier(random_state=11)
lr_clf = LogisticRegression(random_state=11)


# In[15]:


# train과 validation을 통해서 미리
# 학습된 알고리즘 및 가장 높은 정확도의 알고리즘 선택

## dt_clf 학습
dt_clf.fit(X_train, y_train)
pred_dt = dt_clf.predict(X_test)
accuracy_dt = accuracy_score(y_test,pred_dt)

print('dt_clf의 정확도:', np.round(accuracy_dt,4))

## rf_clf 학습
rf_clf.fit(X_train,y_train)
pred_rf = rf_clf.predict(X_test)
accuracy_rf = accuracy_score(y_test,pred_rf)

print('rf_clf의 정확도:', np.round(accuracy_rf,4))

## lr_clf 학습
lr_clf.fit(X_train,y_train)
pred_lr = lr_clf.predict(X_test)
accuracy_lr = accuracy_score(y_test,pred_lr)

print('lr_clf의 정확도:', np.round(accuracy_lr,4))


# In[16]:


from sklearn.model_selection import GridSearchCV

parameters = {'max_depth':[2,3,5,10],
              'min_samples_split':[2,3,5],
              'min_samples_leaf':[1,5,8]}


# In[17]:


grid_dt_clf = GridSearchCV(dt_clf, param_grid=parameters, scoring='accuracy', cv=5)
grid_dt_clf.fit(X_train,y_train) # 전체 train이 아닌 validation을 분할하고
                                 # 난 뒤의 train입니다.

print('grid_dt_clf 최적 파라미터:', grid_dt_clf.best_params_)
print('grid_dt_clf 최고 정확도:', np.round(grid_dt_clf.best_score_,4))


# In[18]:


import time
from sklearn.metrics import accuracy_score

# 분류 알고리즘
start_time = time.time()

rf_clf = RandomForestClassifier()

rf_clf.fit(X_train,y_train)
pred_rf = rf_clf.predict(X_test)
print('Randomforest 정확도:{0:.4f}'.format(accuracy_score(y_test,pred_rf)))
print('Randomforest 수행시간:{0:.2f}초'.format(time.time()-start_time))


# In[19]:


from sklearn.model_selection import RandomizedSearchCV
get_ipython().run_line_magic('pinfo', 'RandomizedSearchCV')


# In[17]:


## end of preprocessing


# In[18]:


## we will start to analyze 

