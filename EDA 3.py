#!/usr/bin/env python
# coding: utf-8

# Flight price Prediction (EDA+ Feature Engineering)

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


train_df= pd.read_excel('Data_Train.xlsx')


# In[3]:


train_df.head()


# In[4]:


test_df= pd.read_excel('Test_set.xlsx')


# In[5]:


test_df.head()


# In[6]:


final_df = train_df.append(test_df)
final_df.head()


# In[7]:


final_df.info()


# In[8]:


##Feature Engineering Process
final_df['Date']=final_df['Date_of_Journey'].str.split('/').str[0]
final_df['Month']=final_df['Date_of_Journey'].str.split('/').str[1]
final_df['Year']=final_df['Date_of_Journey'].str.split('/').str[2]


# In[9]:


final_df.head(2)


# In[10]:


final_df['Date'] = final_df['Date'].astype(int)
final_df['Month'] = final_df['Month'].astype(int)
final_df['Year'] = final_df['Year'].astype(int)


# In[11]:


final_df.info()


# In[12]:


final_df.drop('Date_of_Journey', axis =1 , inplace = True)


# In[13]:


final_df.head(1)


# In[14]:


final_df['Arrival_Time']=final_df['Arrival_Time'].str.split(' ').str[0]


# In[15]:


final_df.head()


# In[16]:


final_df.isnull().sum()


# In[17]:


final_df['Arrival_hour']=final_df['Arrival_Time'].str.split(':').str[0]
final_df['Arrival_min']=final_df['Arrival_Time'].str.split(':').str[1]


# In[18]:


final_df['Arrival_hour']= final_df['Arrival_hour'].astype(int)
final_df['Arrival_min']= final_df['Arrival_min'].astype(int)


# In[19]:


final_df.info()


# In[20]:


final_df.drop('Arrival_Time', axis=1, inplace= True)


# In[21]:


final_df.head(1)


# In[22]:


final_df['Dept_hour']=final_df['Dep_Time'].str.split(':').str[0]
final_df['Dept_min']=final_df['Dep_Time'].str.split(':').str[0]
final_df['Dept_hour']= final_df['Dept_hour'].astype(int)
final_df['Dept_min']= final_df['Dept_min'].astype(int)
final_df.drop('Dep_Time', axis=1, inplace= True)


# In[23]:


final_df.info()


# In[24]:


final_df['Total_Stops'].unique()


# In[25]:


final_df['Total_Stops']


# In[26]:


final_df[final_df['Total_Stops'].isnull()]


# In[27]:


final_df['Total_Stops']=final_df['Total_Stops'].map({'non-stop':0,'1 stop':1,'2 stops':2,'3 stops':3,'4 stops':4,'nan':1})
final_df.drop('Route', axis = 1, inplace=True)
final_df.head()


# In[28]:


final_df['Additional_Info'].unique()


# In[29]:


final_df.info()


# In[30]:


final_df['duration_hour']=final_df['Duration'].str.split(' ').str[0].str.split('h').str[0]


# In[31]:


final_df[final_df['duration_hour']=='5m']


# In[32]:


final_df.drop(6474, axis =0, inplace =True)
final_df.drop(2660, axis =0, inplace =True)


# In[33]:


final_df['duration_hour']= final_df['duration_hour'].astype('int')


# In[34]:


final_df


# In[35]:


final_df.drop('Duration', axis =1, inplace= True)


# In[36]:


final_df.head(1)


# In[37]:


final_df['Airline'].unique()


# In[38]:


from sklearn.preprocessing import LabelEncoder


# In[39]:


labelencoder = LabelEncoder()


# In[40]:


final_df['Airline']= labelencoder.fit_transform(final_df['Airline'])
final_df['Source']= labelencoder.fit_transform(final_df['Source'])
final_df['Destination']= labelencoder.fit_transform(final_df['Destination'])
final_df['Additional_Info']= labelencoder.fit_transform(final_df['Additional_Info'])


# In[41]:


final_df.shape


# In[45]:


final_df[['Airline']]


# In[ ]:


ohe.fit_transform(np.array(final_df['Airline']).reshape(-1,1))


# In[ ]:





# In[ ]:





# In[42]:


from sklearn.preprocessing import OneHotEncoder


# In[46]:


ohe = OneHotEncoder()


# In[47]:


ohe.fit_transform(np.array(final_df['Airline']).reshape(-1,1))

#final_df['Airline']= ohe.fit_transform(final_df['Airline'])
#final_df['Source']= ohe.fit_transform(final_df['Source'])
#final_df['Destination']= ohe.fit_transform(final_df['Destination'])
#final_df['Additional_Info']= ohe.fit_transform(final_df['Additional_Info'])


# In[48]:



final_df.head()


# In[50]:


pd.get_dummies(final_df,columns=["Airline", "Source", "Destination"] ,drop_first = True)


# In[51]:


final_df.info()


# In[55]:


pd.get_dummies(final_df['Airline'])
pd.get_dummies(final_df['Source'])




#pd.get_dummies(final_df,columns=["Airline", "Source", "Destination"] ,drop_first = True)


# In[ ]:




