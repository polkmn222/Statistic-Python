#!/usr/bin/env python
# coding: utf-8

# In[1]:


### 요구사항 1. 알고리즘 3개 정도 - randomforest, decisionTree, LogisticReression
###          2. 평가지표? 2진분류입니다 ^^ - accuracy_score, precision_score, recall_score
###          3.                              f1_score, confusion_matrix, roc_auc_score

# 예 from sklearn.metrics import 


# In[2]:


### 전처리 :: 문자 (X), 결측치 (X)
### target --> outcome 


### describe 및 info 


# * Pregnancies: 임신횟수
# 
# 
# * Glucose: 포도당 부하 검사 수치
# 
# 
# * BloodPressure: 혈압
# 
# 
# * SkinThickness: 팔 삼두근 뒤쪽의 피하지방 측정값(mm)
# 
# 
# * Insulin: 혈청 인슐린
# 
# 
# * BMI: 체질량지수(체중(kg)/키(m))^2)
# 
# 
# * DiabetesPedigreeFunction: 당뇨 내력 가중치값
# 
# 
# * Age: 나이
# 
# 
# * Outcome: 클래스 결정 값(0또는 1)

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[4]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import *


# In[5]:


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

diabetes_df = pd.read_csv('./diabetes.csv')


# In[6]:


diabetes_df.info()


# In[7]:


diabetes_df.describe()


# In[8]:


diabetes_df['Outcome'].value_counts()


# In[26]:


def get_con_index(y_test,pred):
    confusion = confusion_matrix(y_test,pred)
    accuracy = accuracy_score(y_test,pred)
    p_score = precision_score(y_test,pred)
    r_score = recall_score(y_test,pred)
    roc_auc = roc_auc_score(y_test,pred)
    f1 = f1_score(y_test,pred)
    print('혼동행렬 confusion matrix')
    print(confusion)
    print('accuracy :{0:.4f}, precision: {1:.4f}, recall: {2:.4f}, f1:{3:.4f}, roc_auc:{4:.4f}'.format(accuracy, p_score, r_score, f1, roc_auc))


# In[10]:


y = diabetes_df.Outcome
X = diabetes_df.drop(['Outcome'], axis=1)


# In[27]:


import warnings
warnings.filterwarnings('ignore')

# train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, stratify=y)

# fitting 시작

# from sklearn.linear_model import LogisticRegression
lr_clf = LogisticRegression()
lr_clf.fit(X_train,y_train)
pred_lr = lr_clf.predict(X_test)
print('## 로지스틱회귀 ##\n')
get_con_index(y_test,pred_lr)


# from sklearn.tree import DecisionTreeClassifier
dt_clf = DecisionTreeClassifier()
dt_clf.fit(X_train,y_train)
pred_dt = dt_clf.predict(X_test)
print('\n## 의사결정나무 ##\n')
get_con_index(y_test,pred_dt)


# from sklearn.ensemble import RandomForestClassifier
rf_clf = RandomForestClassifier()
rf_clf.fit(X_train,y_train)
pred_rf = rf_clf.predict(X_test)

print('\n## 랜덤포레스트 ##\n')
get_con_index(y_test,pred_rf)


# In[12]:


# matplotlib의 학습 이후에 코딩을 실습시 확인해주셔요 ^^

def p_r_curve_plot(y_test, pred_proba):
    # threshold ndarray로 가져와보고 
    # 위의 threshold에 따른 정밀도, 재현율의 ndarray를 추출
    precisions, recalls, thresholds = precision_recall_curve(y_test, pred_proba)
    
    # X축 = threshold
    # y축은 precision(점선) 및 recall(주황선) 설정
    # 각 곡선을 중첩되게 graph화
    plt.figure(figsize=(8,6))
    threshold_boudary = thresholds.shape[0]
    plt.plot(thresholds, precisions[0:threshold_boudary], linestyle='--', label='precison')
    plt.plot(thresholds, recalls[0:threshold_boudary], label='recall')
    
    # threshold 값 x 축의 scale을 0,1으로 scaling
    start, end = plt.xlim()
    plt.xticks(np.round(np.arange(start,end,0.10),2))
    
    # X축, y축 label 및 legend 추가
    # grid도 추가
    plt.xlabel('Threshold_value')
    plt.ylabel('Precision and Recall')
    plt.legend()
    plt.grid()
    plt.show() 


# In[13]:


lr_clf.predict(X_test)[:10]


# In[14]:


pred_proba = lr_clf.predict_proba(X_test)[:,1]
p_r_curve_plot(y_test,pred_proba)


# In[15]:


test_array = np.arange(1,11)
test_array.reshape(-1,2)[:,:]


# In[16]:


diabetes_df.describe()


# In[17]:


# 혈압에 0이 나온다???

plt.hist(diabetes_df.BloodPressure)
plt.show()


# In[18]:


# Glucose에 0이 나온다???

plt.hist(diabetes_df.Glucose)
plt.show()


# In[19]:


# 0값이 있는 비율을 계산하여 확인해보자.
# Glucose	BloodPressure	SkinThickness	Insulin	BMI

zero_ftrs = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']

total_cnt = len(diabetes_df)

for ftr in zero_ftrs:
    z_cnt = diabetes_df[diabetes_df[ftr]==0][ftr].count()
    print('{0}의 zero의 수는 {1}, 퍼센트 비율은{2:.2f} %'.format(ftr,z_cnt,(z_cnt/total_cnt)*100))


# In[20]:


mean_zero_ftrs = diabetes_df[zero_ftrs].mean()
diabetes_df[zero_ftrs] = diabetes_df[zero_ftrs].replace(0,mean_zero_ftrs)


# In[21]:


X_scaled = diabetes_df.drop(['Outcome'], axis=1)
y_scaled = diabetes_df['Outcome']


# In[22]:


# 스케일링된 데이터를 통해 train_test_split을 함

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, stratify=y_scaled)

print('X_train의 shape:', X_train.shape)
print('X_test의 shape:', X_test.shape)
print('y_train의 shape:', y_train.shape)
print('y_test의 shape:', y_test.shape)


# In[23]:


# fitting 시작

# from sklearn.linear_model import LogisticRegression
lr_clf = LogisticRegression()
lr_clf.fit(X_train,y_train)
pred_lr = lr_clf.predict(X_test)
print('## 로지스틱회귀 ##\n')
get_con_index(y_test,pred_lr)


# from sklearn.tree import DecisionTreeClassifier
dt_clf = DecisionTreeClassifier()
dt_clf.fit(X_train,y_train)
pred_dt = dt_clf.predict(X_test)
print('\n## 의사결정나무 ##\n')
get_con_index(y_test,pred_dt)


# from sklearn.ensemble import RandomForestClassifier
rf_clf = RandomForestClassifier()
rf_clf.fit(X_train,y_train)
pred_rf = rf_clf.predict(X_test)

print('\n## 랜덤포레스트 ##\n')
get_con_index(y_test,pred_rf)


# In[24]:


# end of file

