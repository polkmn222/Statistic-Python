#!/usr/bin/env python
# coding: utf-8

# # Human Activity Recog

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[25]:


ftr_name_df = pd.read_csv('./HAPT Data Set/features.txt', sep='\s+' ,header=None, names=['column_name'])


# In[16]:


ftr_name = ftr_name_df.iloc[:,0].values.tolist()
print('X_ftrs의 data shape:',ftr_name_df.shape)
print('10개의 X_ftrs의 이름:', ftr_name[:10])


# In[19]:


# X_test_df = pd.read_csv('./HAPT Data Set/features.txt', header=None, sep='\s+', names=ftr_name)


# In[23]:


# step 1
cum_cnt_ftr_name = ftr_name_df.groupby(by='column_name').cumcount()

# step 2
pd.DataFrame(cum_cnt_ftr_name,columns= ['copy_cnt'])

# step 3
new_ftr_df = pd.DataFrame(ftr_name_df.groupby(by='column_name').cumcount(), columns=['copy_cnt'])
new_ftr_df= new_ftr_df.reset_index()
new_ftr_df

# step 4 원래의 컬럼명을 갖고 있는 아래의 데이터에
# reset_index를 통해 새로운 컬럼을 생성해준다.

ftr_name_df = ftr_name_df.reset_index()

print(new_ftr_df.columns)
print(ftr_name_df.columns)

ftr_name_df = pd.merge(ftr_name_df,new_ftr_df, how='outer')
ftr_name_df[ftr_name_df['copy_cnt']>0]


# In[24]:


### 참고 cumcount() API

test_df = pd.DataFrame({'ysp':['A','B','C','C','C','D']})
print(test_df)
test_df['cum_count']= test_df.groupby(by='ysp').cumcount()
test_df


# In[26]:


# end of file

